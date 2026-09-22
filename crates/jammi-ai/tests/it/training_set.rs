//! The training set as a producer output — the `jammi-ai` half.
//!
//! A tabular fine-tune does not re-run its source query into memory: it
//! materialises the projected rows into an immutable `TrainingSet` result table
//! through `ResultStore::materialize_training_set`, then reads that table back
//! with the producer's own canonical `ORDER BY` re-applied. These tests pin the
//! four properties that change hands at that seam.

use std::collections::BTreeMap;
use std::sync::Arc;

use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

/// FNV-1a over a byte slice, as `{len}:{hash:016x}`.
///
/// A self-contained fingerprint: `sha2` is an optional *library* dependency of
/// this crate (behind `local`) and not a dev-dependency, so a test target
/// cannot name it, and `DefaultHasher` is explicitly not stable across
/// toolchains — neither can back a constant pinned in source. FNV-1a is fully
/// specified, so the pinned constants below mean the same thing on every host
/// and every toolchain, and the length is carried alongside the hash so a
/// truncation cannot hide behind a collision.
fn fingerprint(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{}:{:016x}", bytes.len(), hash)
}

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// The parity fixture's config — the smallest deterministic job-path
/// fine-tune: one epoch over the 30-row `training_pairs.csv` contrastive
/// source, rank-4 LoRA, no warm-up, the default (constant) seed.
fn parity_config() -> FineTuneConfig {
    FineTuneConfig {
        epochs: 1,
        batch_size: 8,
        lora_rank: 4,
        warmup_steps: 0,
        ..Default::default()
    }
}

fn parity_columns() -> Vec<String> {
    vec![
        "text_a".to_string(),
        "text_b".to_string(),
        "score".to_string(),
    ]
}

async fn session_over(dir: &TempDir, csv_fixture: &str) -> Arc<InferenceSession> {
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(csv_fixture.to_string()),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
}

/// Run the parity fixture to completion and return every published adapter
/// file's fingerprint, keyed by file name (so a file appearing or disappearing
/// moves the oracle as loudly as a byte change does).
/// Fingerprint every file the published model prefix holds, keyed by file
/// name — the print set the byte-for-byte pinned fixtures in this file
/// compare against. One reader for every pin, so the exclusion below is
/// applied once, never re-derived per fixture.
///
/// `materialization.json` embeds `produced_at` (wall-clock) and `produced_by`
/// (a per-process run id) — never byte-stable across runs by design
/// (provenance metadata, not the reproducibility anchor; see
/// `MaterializationManifest`'s own doc), so it can never join a
/// byte-for-byte pinned fixture the way the other files here can. Excluded
/// from the print set rather than pinned or ignored silently: this comment
/// is the record of why the file every fine-tune now publishes is absent
/// from each `*_ADAPTER_PRINTS` fixture.
fn pinned_prints(dir: &std::path::Path) -> BTreeMap<String, String> {
    let mut prints = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == "materialization.json" {
                continue;
            }
            prints.insert(name, fingerprint(&std::fs::read(entry.path()).unwrap()));
        }
    }
    prints
}

async fn run_parity_fixture(session: &Arc<InferenceSession>) -> BTreeMap<String, String> {
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(session)
        .expect("default worker intervals are valid");
    let job = session
        .fine_tune(
            "training",
            &tiny_bert_model(),
            &parity_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(parity_config()),
        )
        .await
        .unwrap();
    let job_id = job.job_id.clone();
    job.wait().await.unwrap();

    // The pinned adapter prints below must be produced by the
    // STREAMED path, not a silently-unchanged eager one — a `test-hooks`
    // observation of the source kind `run_spec` actually bound for this job
    // (mining/GradCache are off in `parity_config`, so `whole_set_arm` is
    // `None` and the worker must have selected `Streamed`).
    assert_eq!(
        jammi_ai::fine_tune::worker::training_test_hooks::source_kind_for(&job_id),
        Some("streamed"),
        "refactor_parity must run through the Streamed source (no mining, no GradCache)"
    );

    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the fine-tune registers its output model");
    let prefix = crate::common::served_bundle_url(ft);
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix)
        .await
        .expect("the published adapter fetches and verifies");

    pinned_prints(local.dir())
}

/// (c) Refactor parity — a PINNED oracle.
///
/// Every file the smallest deterministic job-path fine-tune publishes,
/// fingerprinted as it was when the trainer read its rows straight out of the
/// source query.
///
/// Routing the rows through an immutable Parquet table instead of straight out
/// of the source query changes *where* the trainer's rows come from; it must
/// not change *which* rows, in *what* order, so it must not change one adapter
/// byte. This fixture carries no NULLs, so the producer's `NULLS FIRST` order
/// key and a plain source read's default NULL placement agree on it — the parity
/// claim is over row order and row content, not over NULL placement.
///
/// **What this pin does NOT cover.** Only the TABULAR arm
/// (`training_set::materialize_projection`) is fingerprinted here. The graph
/// arm never goes through `training_set` at all: it samples in memory and
/// trains directly (it has no training-set table of its own), with its own
/// byte-identity pin at `graph_finetune::fine_tune_graph_end_to_end_completes`.
/// The `NULLS FIRST` order key is likewise stated as
/// intended in `training_set`'s module docs, not pinned by (c), which carries
/// no NULLs to exercise it.
///
/// The per-step `checkpoint_N` files are pinned alongside the final adapter on
/// purpose: they fingerprint the *trajectory*, so a row-order change that a
/// converged final adapter might wash out still moves `checkpoint_1`. All eight
/// files were byte-stable across repeated runs before being pinned — a
/// fingerprint that drifts run-to-run would make this oracle noise, not a pin.
/// The bytes are a function of the CPU architecture — the trainer's float
/// kernels differ between x86_64 and aarch64 — and not of the operating
/// system: aarch64 Linux reproduces aarch64 macOS byte for byte. So every pin
/// in this file is keyed on `target_arch`. A third architecture has no pin and
/// does not compile here, which is the point: it must be measured, not guessed.
#[cfg(target_arch = "x86_64")]
const PARITY_ADAPTER_PRINTS: &[(&str, &str)] = &[
    ("adapter.safetensors", "1184:9cf8a5b1ceb712a7"),
    ("adapter_config.json", "143:1feeeb6239c3fd30"),
    ("checkpoint_1.safetensors", "1184:48dbcc8330661413"),
    ("checkpoint_2.safetensors", "1184:e74ad321104e04cd"),
    ("checkpoint_3.safetensors", "1184:a03cc747ccf863d7"),
    ("checkpoint_4.safetensors", "1184:9cf8a5b1ceb712a7"),
    ("checkpoint_best.safetensors", "1184:9cf8a5b1ceb712a7"),
    ("manifest.json", "788:0f58fbd7e7d09f38"),
];
#[cfg(target_arch = "aarch64")]
const PARITY_ADAPTER_PRINTS: &[(&str, &str)] = &[
    ("adapter.safetensors", "1184:5406ba859233d205"),
    ("adapter_config.json", "143:1feeeb6239c3fd30"),
    ("checkpoint_1.safetensors", "1184:fdee6c194f461948"),
    ("checkpoint_2.safetensors", "1184:164dc509698202fd"),
    ("checkpoint_3.safetensors", "1184:ed0bcf37523d6e03"),
    ("checkpoint_4.safetensors", "1184:5406ba859233d205"),
    ("checkpoint_best.safetensors", "1184:5406ba859233d205"),
    ("manifest.json", "788:a58029cc6cc6ef94"),
];

#[tokio::test(flavor = "multi_thread")]
async fn refactor_parity() {
    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let prints = run_parity_fixture(&session).await;
    println!("PARITY_ADAPTER_PRINTS = {prints:#?}");

    let expected: BTreeMap<String, String> = PARITY_ADAPTER_PRINTS
        .iter()
        .map(|(n, p)| ((*n).to_string(), (*p).to_string()))
        .collect();
    assert_eq!(
        prints, expected,
        "the adapter bytes moved: routing the training rows through the \
         TrainingSet producer must not change which rows the trainer sees, in \
         which order"
    );
}

/// Refactor parity, the regression shape — the fixture
/// extends [`refactor_parity`] (which only covers the contrastive shape)
/// with the SAME pinned-oracle discipline over `task=regression`.
///
/// Pinned from the eager `TrainingSet` read-back, run twice to confirm
/// byte-stability before pinning.
///
/// The regression fixture routes the target scaler (`TrainingDataLoader::
/// regression_targets`) and the `Regression` `TextChunk` decode
/// (`worker::build_training_data_loader`'s regression arm) through paths
/// [`refactor_parity`]'s contrastive fixture never exercises at all.
///
/// Platform-specific like [`PARITY_ADAPTER_PRINTS`]: the Linux set is what the
/// CI hermetic lane produces for this fixture (every CI run re-verifies it, a
/// drifting print failing there), the other set is from an Apple Silicon
/// host.
#[cfg(target_arch = "x86_64")]
const REGRESSION_PARITY_ADAPTER_PRINTS: &[(&str, &str)] = &[
    ("adapter.safetensors", "1888:ab17f8201a9c7603"),
    ("adapter_config.json", "284:6d66bd5b8594e1fa"),
    ("checkpoint_1.safetensors", "1888:d739276311b186e0"),
    ("checkpoint_2.safetensors", "1888:2db5d9cd5ad33788"),
    ("checkpoint_3.safetensors", "1888:ab17f8201a9c7603"),
    ("checkpoint_best.safetensors", "1888:ab17f8201a9c7603"),
    ("manifest.json", "676:7497755381d07072"),
];
#[cfg(target_arch = "aarch64")]
const REGRESSION_PARITY_ADAPTER_PRINTS: &[(&str, &str)] = &[
    ("adapter.safetensors", "1888:247d938373b8756d"),
    ("adapter_config.json", "284:6d66bd5b8594e1fa"),
    ("checkpoint_1.safetensors", "1888:6ca7a223b9045945"),
    ("checkpoint_2.safetensors", "1888:d84d3efd599e933d"),
    ("checkpoint_3.safetensors", "1888:247d938373b8756d"),
    ("checkpoint_best.safetensors", "1888:247d938373b8756d"),
    ("manifest.json", "676:56ce2564022c6625"),
];

fn regression_parity_columns() -> Vec<String> {
    vec!["text".to_string(), "target".to_string()]
}

fn regression_parity_config() -> FineTuneConfig {
    FineTuneConfig {
        epochs: 1,
        batch_size: 8,
        lora_rank: 4,
        warmup_steps: 0,
        ..Default::default()
    }
}

async fn run_regression_parity_fixture(
    session: &Arc<InferenceSession>,
) -> BTreeMap<String, String> {
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(session)
        .expect("default worker intervals are valid");
    let job = session
        .fine_tune(
            "training",
            &tiny_bert_model(),
            &regression_parity_columns(),
            FineTuneMethod::Lora,
            ModelTask::Regression,
            Some(regression_parity_config()),
        )
        .await
        .unwrap();
    let job_id = job.job_id.clone();
    job.wait().await.unwrap();

    // The target scaler and the pinned prints below must
    // both come from the STREAMED path.
    assert_eq!(
        jammi_ai::fine_tune::worker::training_test_hooks::source_kind_for(&job_id),
        Some("streamed"),
        "regression_refactor_parity must run through the Streamed source"
    );

    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the fine-tune registers its output model");
    let prefix = crate::common::served_bundle_url(ft);
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix)
        .await
        .expect("the published adapter fetches and verifies");

    let prints = pinned_prints(local.dir());
    prints
}

#[tokio::test(flavor = "multi_thread")]
async fn regression_refactor_parity() {
    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("regression_years.csv")).await;
    let prints = run_regression_parity_fixture(&session).await;
    println!("REGRESSION_PARITY_ADAPTER_PRINTS = {prints:#?}");

    let expected: BTreeMap<String, String> = REGRESSION_PARITY_ADAPTER_PRINTS
        .iter()
        .map(|(n, p)| ((*n).to_string(), (*p).to_string()))
        .collect();
    assert_eq!(
        prints, expected,
        "the adapter bytes moved: which rows the trainer sees, in which \
         order, and how the target scaler is computed must stay unchanged"
    );
}

/// GradCache (`FineTuneConfig::cached = true`) at `W=1`
/// on the eager `TextRows` path, digest-pinned like [`refactor_parity`]'s
/// adapter bytes (this file's own FNV-1a [`fingerprint`], for the reason
/// stated there: neither `sha2` nor `DefaultHasher` can back a constant
/// pinned in source).
///
/// The fixture: a `Pairs`-format projection — `anchor, positive` only — of
/// the 15-row `training_triplets.csv` source, one epoch, rank-4 LoRA,
/// GradCache on, `MultipleNegativesRanking` at temperature 20. A streaming
/// loader over the same fixture produces BYTE-IDENTICAL digests (chunk-for-
/// chunk parity), so the same pinned bytes witness the eager path. Confirmed
/// byte-stable across two repeated runs before being pinned.
#[tokio::test(flavor = "multi_thread")]
async fn gradcache_completes_at_w1_with_a_pinned_adapter_digest() {
    use jammi_ai::fine_tune::EmbeddingLoss;

    // Platform-specific like `PARITY_ADAPTER_PRINTS`: the Linux set from the
    // CI hermetic lane, the other from an Apple Silicon host.
    #[cfg(target_arch = "x86_64")]
    const GRADCACHE_ADAPTER_PRINTS: &[(&str, &str)] = &[
        ("adapter.safetensors", "1184:fa6bd33abdb851b5"),
        ("adapter_config.json", "143:1feeeb6239c3fd30"),
        ("checkpoint_1.safetensors", "1184:fa6bd33abdb851b5"),
        ("checkpoint_best.safetensors", "1184:fa6bd33abdb851b5"),
        ("manifest.json", "452:d47c82d1dbae175e"),
    ];
    #[cfg(target_arch = "aarch64")]
    const GRADCACHE_ADAPTER_PRINTS: &[(&str, &str)] = &[
        ("adapter.safetensors", "1184:06234c2bf6312b14"),
        ("adapter_config.json", "143:1feeeb6239c3fd30"),
        ("checkpoint_1.safetensors", "1184:06234c2bf6312b14"),
        ("checkpoint_best.safetensors", "1184:06234c2bf6312b14"),
        ("manifest.json", "452:ba02b892c5f53429"),
    ];

    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_triplets.csv")).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "training",
            &tiny_bert_model(),
            &["anchor".to_string(), "positive".to_string()],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 4,
                lora_rank: 4,
                warmup_steps: 0,
                cached: true,
                embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let job_id = job.job_id.clone();
    job.wait()
        .await
        .expect("a W=1 GradCache run over the eager loader must complete");

    // `whole_set_arm` must select `Resident` for a
    // GradCache-eligible configuration — the complementary oracle to
    // `refactor_parity`/`regression_refactor_parity`'s `Streamed`
    // observation above (mining on / GradCache on → `Resident`, never
    // `Streamed`).
    assert_eq!(
        jammi_ai::fine_tune::worker::training_test_hooks::source_kind_for(&job_id),
        Some("resident"),
        "a GradCache-eligible run must bind Resident, never Streamed"
    );

    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the GradCache run registers its output model");
    let prefix = crate::common::served_bundle_url(ft);
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix)
        .await
        .expect("the published GradCache adapter fetches and verifies");

    let prints = pinned_prints(local.dir());
    println!("GRADCACHE_ADAPTER_PRINTS = {prints:#?}");
    let expected: BTreeMap<String, String> = GRADCACHE_ADAPTER_PRINTS
        .iter()
        .map(|(n, p)| ((*n).to_string(), (*p).to_string()))
        .collect();
    assert_eq!(
        prints, expected,
        "the GradCache adapter bytes moved from the pinned fixture"
    );
}

/// The mining W=1 byte-parity oracle, with its non-vacuity control: a test
/// that sets `hard_negatives.mine = true` with NO `embedding_loss` at all
/// measures nothing, because `mining_eligible()` (which ALSO requires the
/// in-batch-negative objective, `source.rs`'s own doc) never admits mining,
/// and flipping `mine` off leaves the pinned bytes identical. This test sets an `embedding_loss`
/// the predicate admits (`MultipleNegativesRanking`) and drives the SAME
/// fixture twice, `mine` the only field that differs, then asserts the
/// trained adapter bytes MOVE.
///
/// **Mining asserted to have run, not merely configured.** `cached: false`
/// on both runs is load-bearing: `whole_set_arm` checks `mining_eligible`
/// BEFORE `gradcache_eligible` (`source.rs::whole_set_arm`'s own doc), but a
/// bare `Resident` source-kind observation is ambiguous between the two
/// arms when both COULD be eligible — no test hook distinguishing them
/// directly exists. Pinning
/// `cached: false` removes that ambiguity structurally instead: with
/// GradCache ineligible on this config, `Resident` cannot be explained by
/// anything other than `WholeSetArm::Mining`.
///
/// **The byte-for-byte pin, GradCache-shaped.**
/// Like [`gradcache_completes_at_w1_with_a_pinned_adapter_digest`],
/// this crate's CPU backprop is demonstrably not byte-identical across
/// CPU architectures (a measured divergence on this very fixture, not a
/// hypothetical one), so the pin is a per-`target_arch` constant pair, never
/// one shared value. The aarch64 constant below is pinned from two repeated
/// local runs (confirmed byte-stable, the same discipline
/// [`gradcache_completes_at_w1_with_a_pinned_adapter_digest`]'s own doc
/// states). The Linux constant is pinned from two agreeing captures on
/// x86_64 Linux (the hermetic CI job and the CI image on another host), read
/// from the `println!` below, printed BEFORE the assert on every run; every
/// CI run re-measures it and a divergence fails BY NAME here.
/// This run's live mining-on-vs-off inequality (below) is the non-vacuity
/// control described above.
#[tokio::test(flavor = "multi_thread")]
async fn hard_negative_mining_at_w1_moves_the_adapter_bytes_mining_off_leaves_it_unreached() {
    use jammi_ai::fine_tune::{EmbeddingLoss, HardNegativeConfig};

    // Platform-specific, like `PARITY_ADAPTER_PRINTS` above: the Linux pair
    // is captured from the hermetic CI job's stdout (this test's own
    // `println!`, below) and confirmed by a second run in the CI image on
    // another x86_64 host; the macOS pair is pinned from two repeated local
    // runs.
    #[cfg(target_arch = "x86_64")]
    const MINING_ADAPTER_PRINTS: &[(&str, &str)] = &[
        ("adapter.safetensors", "1184:442a2a1013d43364"),
        ("adapter_config.json", "143:1feeeb6239c3fd30"),
        ("checkpoint_1.safetensors", "1184:d6efb625ea7bc057"),
        ("checkpoint_2.safetensors", "1184:13630d62b8b2ca78"),
        ("checkpoint_3.safetensors", "1184:c0f5a2deff6bf5b7"),
        ("checkpoint_4.safetensors", "1184:442a2a1013d43364"),
        ("checkpoint_best.safetensors", "1184:442a2a1013d43364"),
        ("manifest.json", "788:fb18b2979128597b"),
    ];
    #[cfg(target_arch = "aarch64")]
    const MINING_ADAPTER_PRINTS: &[(&str, &str)] = &[
        ("adapter.safetensors", "1184:ee37dbdc4de182fc"),
        ("adapter_config.json", "143:1feeeb6239c3fd30"),
        ("checkpoint_1.safetensors", "1184:f73470d7bd26a6d3"),
        ("checkpoint_2.safetensors", "1184:ed090f96d9512427"),
        ("checkpoint_3.safetensors", "1184:a87606f52162f0de"),
        ("checkpoint_4.safetensors", "1184:ee37dbdc4de182fc"),
        ("checkpoint_best.safetensors", "1184:ee37dbdc4de182fc"),
        ("manifest.json", "788:8cc7fb24393325ce"),
    ];

    async fn run(dir: &TempDir, mine: bool) -> (Option<&'static str>, BTreeMap<String, String>) {
        let session = session_over(dir, &common::fixture_url("training_triplets.csv")).await;
        let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
            .expect("default worker intervals are valid");
        let job = session
            .fine_tune(
                "training",
                &tiny_bert_model(),
                &["anchor".to_string(), "positive".to_string()],
                FineTuneMethod::Lora,
                ModelTask::TextEmbedding,
                Some(FineTuneConfig {
                    epochs: 1,
                    batch_size: 4,
                    lora_rank: 4,
                    warmup_steps: 0,
                    cached: false,
                    embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking {
                        temperature: 20.0,
                    }),
                    hard_negatives: HardNegativeConfig {
                        mine,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        let job_id = job.job_id.clone();
        job.wait()
            .await
            .expect("a W=1 mining-config run must complete");

        let source_kind =
            jammi_ai::fine_tune::worker::training_test_hooks::source_kind_for(&job_id);

        let models = session.catalog().list_models().await.unwrap();
        let ft = models
            .iter()
            .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
            .expect("the run registers its output model");
        let prefix = crate::common::served_bundle_url(ft);
        let local = session
            .artifact_store()
            .fetch_artifact(&prefix)
            .await
            .expect("the published adapter fetches and verifies");
        let prints = pinned_prints(local.dir());
        (source_kind, prints)
    }

    let mining_dir = TempDir::new().unwrap();
    let (mining_kind, mining_prints) = run(&mining_dir, true).await;
    assert_eq!(
        mining_kind,
        Some("resident"),
        "mine=true under MultipleNegativesRanking, cached=false, must select Resident \
         (mining_eligible) — cached=false on this same config rules out GradCache as an \
         alternative explanation, so Resident here is proof the Mining arm, specifically, ran"
    );

    // Printed BEFORE the byte-for-byte assert below, unconditionally on every
    // run (Linux included) — a failing test's captured stdout is the value
    // to re-pin from.
    println!("MINING_ADAPTER_PRINTS = {mining_prints:#?}");

    let off_dir = TempDir::new().unwrap();
    let (off_kind, off_prints) = run(&off_dir, false).await;
    assert_eq!(
        off_kind,
        Some("streamed"),
        "mine=false, cached=false leaves whole_set_arm() with no eligible arm at all — the run \
         must take the Streamed path, never Resident, confirming the ONLY difference between the \
         two runs is whether the miner ran"
    );

    let mining_print = mining_prints
        .get("adapter.safetensors")
        .expect("every run publishes adapter.safetensors");
    let off_print = off_prints
        .get("adapter.safetensors")
        .expect("every run publishes adapter.safetensors");
    assert_ne!(
        mining_print, off_print,
        "hard-negative mining must change the trained adapter bytes — an identical adapter print \
         with mining flipped on would mean the miner's replaced (anchor, positive, mined-negative) \
         triplets never reached the trainer. `mining_eligible()` never admits mining without an \
         `embedding_loss` set at all; this run sets one, so a regression that silently dropped the \
         mined loader (falling back to the original triplets) is caught here."
    );

    // The byte-for-byte pin, asserted on both platforms; a divergence fails
    // BY NAME below (the `println!` above is the value to compare against).
    let expected: BTreeMap<String, String> = MINING_ADAPTER_PRINTS
        .iter()
        .map(|(n, p)| ((*n).to_string(), (*p).to_string()))
        .collect();
    assert_eq!(
        mining_prints, expected,
        "the mining-on adapter bytes moved from the pinned fixture"
    );
}

/// (a) A fine-tune job creates a `ready` `TrainingSet` result table carrying a
/// definition hash and a manifest attestation, and trains from it.
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_job_creates_and_trains_from_a_training_set_table() {
    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let prints = run_parity_fixture(&session).await;
    assert!(
        !prints.is_empty(),
        "the job trained and published an adapter"
    );

    let tables = session
        .catalog()
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .await
        .unwrap();
    let training_sets: Vec<_> = tables
        .iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .collect();
    assert_eq!(
        training_sets.len(),
        1,
        "the job materialises exactly one training set, got {:?}",
        tables
            .iter()
            .map(|t| (&t.table_name, &t.kind))
            .collect::<Vec<_>>()
    );
    let table = training_sets[0];
    assert_eq!(table.row_count, 30, "every source row is committed");
    assert!(
        table.definition_hash.is_some(),
        "a producer output is content-addressed by its definition hash"
    );

    let descriptor = session
        .result_store()
        .producing_descriptor(&common::pin(&session, table.clone()).await)
        .await
        .expect("the attestation records the producing descriptor verbatim");
    match descriptor {
        jammi_db::store::manifest::ProducingDescriptor::TrainingSet {
            source,
            columns,
            task,
            format,
            order_rule,
        } => {
            // The recorded source is the query the producer actually ran, with
            // the columns in DECLARED order: the declared order is also the
            // order key, so a descriptor that recorded some other projection
            // order would name a table it does not describe.
            assert_eq!(
                source,
                r#"SELECT "text_a", "text_b", "score" FROM "training".public."training_pairs""#
            );
            assert_eq!(columns, parity_columns());
            assert_eq!(task, ModelTask::TextEmbedding);
            assert_eq!(format, "contrastive");
            assert_eq!(
                order_rule,
                jammi_db::store::manifest::TRAINING_SET_ORDER_RULE_V1
            );
        }
        other => panic!("expected a TrainingSet descriptor, got {other:?}"),
    }
}

/// Corrected (b) — two fine-tune JOBS over the same plain source, columns,
/// task and format materialise TWO training-set tables, never one.
///
/// Two jobs over the same source/columns/task/format do not reuse ONE table:
/// reuse requires pinned EQUAL anchors, and a registered source exposes no
/// version surface, so the engine anchors it
/// [`AnchorKind::UnpinnedAtInstant`](jammi_db::store::manifest::AnchorKind::UnpinnedAtInstant)
/// and never reuses across two independent reads of it — the same honest
/// off-ness the embedding cache records (`pipeline::embedding`).
/// Reuse over a genuinely PINNED anchor is exercised at the store level, in
/// `two_runs_over_one_pinned_definition_share_one_training_set`
/// (`crates/jammi-db/tests/it/materialization.rs`); this is the
/// job-level corollary: each job's OWN materialize call runs the producer
/// fresh (the reuse probe never matches), so two jobs leave two tables
/// behind, each the one its own run actually read from before training.
#[tokio::test(flavor = "multi_thread")]
async fn two_jobs_over_one_plain_source_materialise_two_training_sets() {
    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let mut model_ids = Vec::new();
    for _ in 0..2 {
        let job = session
            .fine_tune(
                "training",
                &tiny_bert_model(),
                &parity_columns(),
                FineTuneMethod::Lora,
                ModelTask::TextEmbedding,
                Some(parity_config()),
            )
            .await
            .unwrap();
        job.wait().await.unwrap();
        model_ids.push(job.model_id().to_string());
    }
    assert_ne!(
        model_ids[0], model_ids[1],
        "two distinct jobs register two distinct output models"
    );
    for model_id in &model_ids {
        let record = session
            .catalog()
            .get_model(model_id)
            .await
            .unwrap()
            .expect("each job registers its own output model");
        assert!(
            record.location.is_some(),
            "job {model_id} must have published an adapter"
        );
    }

    let tables = session
        .catalog()
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .await
        .unwrap();
    let training_sets: Vec<_> = tables
        .iter()
        .filter(|t| t.kind == ResultTableKind::TrainingSet)
        .collect();
    assert_eq!(
        training_sets.len(),
        2,
        "two jobs over an unpinned plain source must materialise TWO tables \
         (never reused), got {:?}",
        training_sets
            .iter()
            .map(|t| &t.table_name)
            .collect::<Vec<_>>()
    );
    assert_ne!(
        training_sets[0].table_name, training_sets[1].table_name,
        "the two tables must have distinct ids"
    );
    for table in &training_sets {
        assert_eq!(
            table.row_count, 30,
            "every source row is committed, per table"
        );
    }

    // The definition hash folds source/columns/task/format/order_rule —
    // NOT the anchor, the table id, or anything about which job ran it. Two
    // tables that are never reused (because their anchors are unpinned and so
    // never compare equal) still name the SAME definition: it is the anchor
    // that differs between them, not the descriptor.
    let hashes: Vec<Option<String>> = training_sets
        .iter()
        .map(|t| t.definition_hash.clone())
        .collect();
    assert!(
        hashes.iter().all(Option::is_some),
        "a producer output is content-addressed by its definition hash, got {hashes:?}"
    );
    assert_eq!(
        hashes[0], hashes[1],
        "two jobs over the identical source/columns/task/format must record ONE \
         definition_hash between their two tables, got {hashes:?}"
    );
}

/// (e) A projection that yields zero rows is refused with the typed
/// `EmptyTrainingSet` before any training set exists, through the JOB path.
///
/// A zero-row training set is the shape that trains silently on nothing: the
/// loop runs, an adapter is published, and every metric is a fold over an empty
/// set. The refusal has to reach the JOB's terminal error, not merely the
/// producer's return value, which is why this drives `fine_tune` end to end
/// rather than calling the verb.
#[tokio::test(flavor = "multi_thread")]
async fn empty_projection_is_refused_through_the_job_path() {
    let dir = TempDir::new().unwrap();
    // A header-only CSV: the schema resolves and the projection is valid, so
    // no column-shape check can catch this — only the row count can.
    let csv = dir.path().join("empty_pairs.csv");
    std::fs::write(&csv, "text_a,text_b,score\n").unwrap();
    let session = session_over(&dir, &format!("file://{}", csv.display())).await;

    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let job = session
        .fine_tune(
            "training",
            &tiny_bert_model(),
            &parity_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            // `train_loss` early stopping so the ONLY thing wrong with this
            // job is that its projection is empty: the `val_loss` default has
            // its own guard against an empty validation split, which would
            // fail the job for an incidental reason and make this oracle
            // vacuous about the emptiness itself.
            Some(FineTuneConfig {
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                ..parity_config()
            }),
        )
        .await
        .unwrap();
    let outcome = job.wait().await;
    let record = session.catalog().get_job(&job.job_id).await.unwrap();
    println!("empty-training-set job outcome = {outcome:?}");
    println!(
        "empty-training-set job status = {:?} error = {:?}",
        record.status, record.error
    );

    assert_eq!(
        record.status, "failed",
        "a zero-row projection must fail the job, never train on nothing"
    );
    let error = record.error.clone().unwrap_or_default();
    assert!(
        error.contains("the projection yielded zero rows"),
        "the failure must be the typed EmptyTrainingSet refusal, got {error:?}"
    );

    for status in [ResultTableStatus::Ready, ResultTableStatus::Building] {
        let tables = session
            .catalog()
            .list_result_tables_by_status(status)
            .await
            .unwrap();
        assert!(
            !tables
                .iter()
                .any(|t| t.kind == ResultTableKind::TrainingSet),
            "the refusal leaves no {status:?} training-set row behind"
        );
    }
}

fn rows_of(batches: &[arrow::array::RecordBatch]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for batch in batches {
        let anchor = string_column(batch, "anchor");
        let positive = string_column(batch, "positive");
        for i in 0..batch.num_rows() {
            out.push((anchor[i].clone(), positive[i].clone()));
        }
    }
    out
}

/// A string column read without assuming which Arrow string type the reader
/// hands back — the parquet reader yields `Utf8View` in some configurations and
/// `Utf8` in others, and a downcast that assumed one would silently read
/// nothing under the other.
fn string_column(batch: &arrow::array::RecordBatch, name: &str) -> Vec<String> {
    use arrow::array::AsArray;
    let column = batch.column_by_name(name).expect("column present");
    match column.data_type() {
        arrow::datatypes::DataType::Utf8View => column
            .as_string_view()
            .iter()
            .map(|v| v.unwrap_or_default().to_string())
            .collect(),
        arrow::datatypes::DataType::LargeUtf8 => column
            .as_string::<i64>()
            .iter()
            .map(|v| v.unwrap_or_default().to_string())
            .collect(),
        _ => column
            .as_string::<i32>()
            .iter()
            .map(|v| v.unwrap_or_default().to_string())
            .collect(),
    }
}

/// (d) The read-back re-applies the canonical `ORDER BY` and matches the
/// committed order on a table with MORE THAN ONE row group, read at
/// `execution_threads > 1` over a scan that is genuinely split across
/// partitions.
///
/// Each of those three conditions is MEASURED in the test, not assumed: the
/// row-group count comes off the Parquet footer, the partition count off the
/// session config, and the file-group count off the physical plan. A fixture
/// that fitted in one row group, a session that planned one partition, or a
/// scan DataFusion kept in one file group would each make this pass while
/// proving nothing.
///
/// The negative control is the mechanism trace: a plain read of the same
/// relation, minted from the bare catalog name with no sort, comes back in a
/// DIFFERENT order, so the scan's sort is what is doing the work — not an
/// accident of how the file happened to be scanned.
#[tokio::test(flavor = "multi_thread")]
async fn read_back_re_applies_the_committed_order_across_row_groups() {
    let dir = TempDir::new().unwrap();

    let mut config = common::test_config(dir.path());
    config.engine.execution_threads =
        std::num::NonZeroUsize::new(4).expect("a positive thread count");
    let partitions = config.engine.execution_threads.get();
    assert!(
        partitions > 1,
        "a scan cannot interleave at one partition, so the control below would \
         be vacuous"
    );
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    // The 70,000-row multi-row-group fixture: ONE builder every streaming
    // oracle calls.
    let fixture = common::multi_row_group_pairs(&session, dir.path(), true).await;
    let table = fixture.table.clone();
    let columns = fixture.columns.clone();

    assert!(
        fixture.row_groups > 1,
        "the order oracle is vacuous on a single row group; the fixture produced {}",
        fixture.row_groups
    );
    let committed = fixture.canonical_order();
    assert_eq!(committed.len(), fixture.written.len());

    // The read-back the worker performs, through the production reader.
    let ctx = session.context();
    let store = session.result_store();
    let batches = table
        .scan(&store, ctx)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows_of(&batches),
        committed,
        "the read-back must reproduce the committed order exactly"
    );

    // The sort names the WHOLE committed key, rendered by the producer's own
    // single source of truth. A prefix of the key re-sorts an already-sorted
    // file into (almost always) the same order, so the row comparison above
    // cannot see that determinant — the plan is where it is visible: its
    // root is the sort, over every projected column, in declared order,
    // ascending, NULLs first.
    let scan = table.scan(&store, ctx).await.unwrap();
    let datafusion::logical_expr::LogicalPlan::Sort(sort) = scan.logical_plan() else {
        panic!(
            "the scan's plan must be rooted in its sort: {}",
            scan.logical_plan()
        );
    };
    // The frame qualifies each key column against the relation it scans;
    // the key is the column NAMES, in declared order, and each one's
    // direction and NULL placement.
    let key: Vec<(String, bool, bool)> = sort
        .expr
        .iter()
        .map(|e| match &e.expr {
            datafusion::logical_expr::Expr::Column(c) => (c.name.clone(), e.asc, e.nulls_first),
            other => panic!("a sort key is a bare column, got {other}"),
        })
        .collect();
    let rendered: Vec<(String, bool, bool)> = jammi_db::store::training_set_sort_exprs(&columns)
        .iter()
        .map(|e| (e.expr.to_string(), e.asc, e.nulls_first))
        .collect();
    assert_eq!(
        key, rendered,
        "the read-back key must carry every projected column, in declared order, ascending, \
         NULLs first"
    );

    // The unordered read a caller could mint from the bare catalog name —
    // an arbitrary registered relation under that relation's own contract,
    // not the handle's — is the negative control.
    let unordered_sql = format!(
        "SELECT * FROM {}",
        jammi_db::store::result_table_relation(table.table_name())
    );

    // The scan really is split: read it off the physical plan rather than
    // trusting the knob above to have taken effect.
    let plan = explain(&session, &unordered_sql).await;
    assert!(
        plan.contains("file_groups={4 groups"),
        "the read-back must be planned over {partitions} file groups for the \
         control to mean anything; the plan is:\n{plan}"
    );

    // Negative control — remove the claimed cause and confirm the result moves.
    let unordered = rows_of(&session.sql(&unordered_sql).await.unwrap());
    assert_eq!(unordered.len(), committed.len());
    assert_ne!(
        unordered, committed,
        "an unordered scan over {partitions} file groups returned committed \
         order anyway, so this oracle cannot distinguish a reader that \
         re-applies the order from one that does not"
    );
}

/// The rendered physical plan of `query`, for assertions about HOW it is read.
async fn explain(session: &Arc<InferenceSession>, query: &str) -> String {
    let batches = session.sql(&format!("EXPLAIN {query}")).await.unwrap();
    arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string()
}

/// The executed probe behind the module header's claim that a fine-tune source
/// can never name a result table (and so can never be anchored by content
/// digest): resolving a source goes through
/// `SessionContext::catalog(source_id)`, and a result table is registered as a
/// BARE table in the default catalog, never as a catalog of its own.
#[tokio::test(flavor = "multi_thread")]
async fn a_result_table_cannot_be_a_fine_tune_source() {
    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let columns = parity_columns();
    let (table, _) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "training",
        &columns,
        ModelTask::TextEmbedding,
        "contrastive",
    )
    .await
    .unwrap();

    // The table is real and readable under its registered name.
    let bound = session
        .sql(&format!(
            "SELECT * FROM {}",
            jammi_db::store::result_table_relation(table.table_name())
        ))
        .await
        .unwrap();
    assert_eq!(bound.iter().map(|b| b.num_rows()).sum::<usize>(), 30);

    // ... and yet neither its catalog name nor its registered name resolves as
    // a fine-tune SOURCE, on either the submit or the materialize path.
    // The registered form is spelled out here: `jammi.{table_name}`.
    for name in [
        table.table_name().to_string(),
        format!("jammi.{}", table.table_name()),
    ] {
        let err = jammi_ai::fine_tune::training_set::materialize_projection(
            &session,
            &name,
            &columns,
            ModelTask::TextEmbedding,
            "contrastive",
        )
        .await
        .expect_err("a result table must not resolve as a fine-tune source");
        assert!(
            format!("{err}").contains("not found"),
            "expected a source-resolution failure for {name:?}, got {err}"
        );
    }
}

/// A `TrainingSet` table's recorded producer replays, byte-identically
/// when the source has not moved, and the replay genuinely RECOMPUTES rather
/// than resolving back to the table it was asked to replay.
///
/// The second half is the load-bearing one: the verb owns its own reuse probe
/// (there is no `CachePolicy` to pass it), so a replay that the probe answered
/// from the catalog would report a recompute that never ran.
#[tokio::test(flavor = "multi_thread")]
async fn a_training_set_replays_from_its_recorded_descriptor() {
    use jammi_ai::pipeline::recompute::Cascade;

    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let columns = parity_columns();
    let (table, _) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "training",
        &columns,
        ModelTask::TextEmbedding,
        "contrastive",
    )
    .await
    .unwrap();

    let before = artifact_digest(&session, table.table_name()).await;
    let report = jammi_ai::Session::new(Arc::clone(&session))
        .recompute(table.table_name(), Cascade::ReportOnly)
        .await
        .unwrap();

    assert_eq!(report.recomputed.len(), 1);
    let replay = &report.recomputed[0];
    assert_eq!(replay.original, table.table_name());
    assert_ne!(
        replay.recomputed,
        table.table_name(),
        "the replay must write a NEW table, not hand back the one it replayed"
    );
    assert_eq!(
        replay.outcome,
        jammi_db::store::CacheOutcome::Computed,
        "an unpinned source anchor never matches the verb's reuse probe, so a \
         replay always recomputes"
    );
    assert_eq!(
        before,
        artifact_digest(&session, &replay.recomputed).await,
        "the descriptor records every determinant, so a replay over unmoved \
         inputs is byte-identical"
    );
}

/// `recompute`'s `TrainingSet` replay re-anchors from the ORIGINAL
/// manifest's recorded relation names, not from the recomputed table's single
/// `source_id` lineage column.
///
/// Nothing about `TrainingSetSpec` restricts `inputs` to one entry — the
/// tabular arm records exactly one today, but the shape is general the same
/// way `pipeline/asof/verb.rs` anchors its spine and facts relations
/// separately, and this fixture exercises it directly by materialising a spec
/// whose `inputs` name two distinct relations. A replay that instead
/// re-derived a single anchor from `table.source_id` would silently collapse
/// the recorded set to one relation; asserting the replay's OWN manifest is
/// the executed check.
#[tokio::test(flavor = "multi_thread")]
async fn recompute_re_anchors_every_recorded_relation() {
    use jammi_ai::pipeline::recompute::Cascade;
    use jammi_db::store::manifest::{AnchorKind, InputAnchor, ProducingDescriptor};
    use jammi_db::store::{TrainingSetInput, TrainingSetSpec};

    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    // A second, distinctly-named source — its content is irrelevant here,
    // only its NAME matters, as a second relation `inputs` will carry.
    session
        .add_source(
            "training_secondary",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let columns = parity_columns();
    let source_sql =
        r#"SELECT "text_a", "text_b", "score" FROM "training".public."training_pairs""#.to_string();
    let now = chrono::Utc::now().to_rfc3339();
    let spec = TrainingSetSpec {
        source_id: "training",
        input: TrainingSetInput::Sql(&source_sql),
        columns: &columns,
        task: ModelTask::TextEmbedding,
        descriptor: ProducingDescriptor::training_set(
            source_sql.clone(),
            columns.clone(),
            ModelTask::TextEmbedding,
            "contrastive",
        ),
        inputs: vec![
            InputAnchor::unpinned_at_instant("training", now.clone()),
            InputAnchor::unpinned_at_instant("training_secondary", now),
        ],
        device: session.compute_device(),
    };
    let table = session
        .result_store()
        .materialize_training_set(session.context(), spec)
        .await
        .unwrap();

    let report = jammi_ai::Session::new(Arc::clone(&session))
        .recompute(table.table_name(), Cascade::ReportOnly)
        .await
        .unwrap();
    assert_eq!(report.recomputed.len(), 1);
    let replay = &report.recomputed[0];

    let replayed_record = session
        .catalog()
        .get_result_table(&replay.recomputed)
        .await
        .unwrap()
        .expect("the replay promoted a new ready table");
    let anchors: Vec<InputAnchor> = serde_json::from_str(
        replayed_record
            .input_anchors_json
            .as_deref()
            .expect("the replay's manifest carries recorded anchors"),
    )
    .unwrap();
    let names: std::collections::BTreeSet<&str> =
        anchors.iter().map(|a| a.source.as_str()).collect();
    assert_eq!(
        names,
        std::collections::BTreeSet::from(["training", "training_secondary"]),
        "the replay must re-anchor every relation the ORIGINAL manifest recorded, \
         not just the recomputed table's source_id, got {anchors:?}"
    );
    assert_eq!(
        anchors.len(),
        2,
        "no relation may be duplicated or dropped, got {anchors:?}"
    );
    for anchor in &anchors {
        assert_eq!(anchor.kind, AnchorKind::UnpinnedAtInstant);
    }
    assert_eq!(
        anchors[0].anchor, anchors[1].anchor,
        "the replay's re-anchor must share ONE fresh instant across every relation, \
         got {anchors:?}"
    );
}

/// A `TrainingSet` descriptor recorded under
/// an `order_rule` this build does not implement is `NotRecomputable`, never a
/// replay guessed under a rule the recorded descriptor does not claim.
///
/// No producer in this build ever WRITES an unimplemented rule, so the only
/// way to exercise the refusal honestly is to corrupt a real manifest's
/// `order_rule` in place (same bytes, same artifact, same anchors — the ONE
/// field this test changes) and drive `recompute` at it.
#[tokio::test(flavor = "multi_thread")]
async fn recompute_refuses_a_training_set_with_an_unknown_order_rule() {
    use jammi_ai::pipeline::recompute::Cascade;
    use jammi_db::error::JammiError;
    use jammi_db::store::manifest::{MaterializationManifest, ProducingDescriptor};

    const UNKNOWN_ORDER_RULE: &str = "full_tuple_v2";

    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let columns = parity_columns();
    let (table, _) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "training",
        &columns,
        ModelTask::TextEmbedding,
        "contrastive",
    )
    .await
    .unwrap();

    let url = jammi_db::storage::StorageUrl::parse(table.parquet_path()).unwrap();
    let mut manifest: MaterializationManifest = session
        .result_store()
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("the producer wrote a manifest sidecar");
    let ProducingDescriptor::TrainingSet { order_rule, .. } = &mut manifest.descriptor else {
        panic!(
            "expected a TrainingSet descriptor, got {:?}",
            manifest.descriptor
        );
    };
    assert_ne!(
        order_rule.as_str(),
        UNKNOWN_ORDER_RULE,
        "the corruption below is vacuous unless it actually changes the rule"
    );
    *order_rule = UNKNOWN_ORDER_RULE.to_string();

    let handle = session.result_store().open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, manifest.to_json_bytes().unwrap().into())
        .await
        .unwrap();

    let err = jammi_ai::Session::new(Arc::clone(&session))
        .recompute(table.table_name(), Cascade::ReportOnly)
        .await
        .expect_err("an unimplemented order_rule must refuse, never replay under a guess");
    match err {
        JammiError::NotRecomputable { table: named } => {
            assert_eq!(named, table.table_name());
        }
        other => panic!("expected NotRecomputable, got {other:?}"),
    }
}

/// This test pins the OUTER `producing_descriptor` guard, not the
/// between-the-two-reads fold: it deletes the sidecar BEFORE calling
/// `recompute` at all, so the refusal it drives is `recompute`'s outer
/// dispatch reading the descriptor through `ResultStore::producing_descriptor`
/// and finding no sidecar there — the same first read every other kind's
/// recompute refuses on. The narrower race — a sidecar that
/// vanishes strictly BETWEEN that descriptor read and
/// `recompute_training_set`'s own anchor read — cannot be constructed by
/// driving this public entry point end to end (the outer guard above already
/// refuses first); that fold is pinned at the function level by
/// `recompute_training_set_refuses_when_its_own_manifest_read_finds_no_sidecar`
/// (`pipeline/recompute.rs`), which calls `recompute_training_set` directly
/// with the sidecar removed in the window this test cannot reach.
///
/// No producer or verb in this build ever tears the sidecar off a `ready`
/// table on its own, so the only way to exercise even the outer refusal
/// honestly is to delete a real manifest sidecar out from under a real table
/// (same Parquet object, same catalog row — the sidecar is the ONE thing this
/// test removes) and drive `recompute` at it.
#[tokio::test(flavor = "multi_thread")]
async fn recompute_refuses_a_training_set_with_a_missing_sidecar() {
    use jammi_ai::pipeline::recompute::Cascade;
    use jammi_db::error::JammiError;
    use jammi_db::storage::DeleteOutcome;

    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let columns = parity_columns();
    let (table, _) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "training",
        &columns,
        ModelTask::TextEmbedding,
        "contrastive",
    )
    .await
    .unwrap();

    let url = jammi_db::storage::StorageUrl::parse(table.parquet_path()).unwrap();
    let handle = session.result_store().open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    assert!(
        handle.exists(&sidecar).await.unwrap(),
        "the producer must have written a manifest sidecar for the corruption \
         below to mean anything"
    );
    assert_eq!(
        handle.vanish_for_test(&sidecar).await.unwrap(),
        DeleteOutcome::Deleted,
        "the sidecar must actually be removed, or the refusal below would be \
         exercising something else"
    );
    assert!(
        session
            .result_store()
            .read_materialization_manifest(&url)
            .await
            .unwrap()
            .is_none(),
        "with the sidecar gone, the manifest read must report Ok(None), not an error"
    );

    let err = jammi_ai::Session::new(Arc::clone(&session))
        .recompute(table.table_name(), Cascade::ReportOnly)
        .await
        .expect_err("a missing sidecar must refuse, never replay with zero recorded anchors");
    match err {
        JammiError::NotRecomputable { table: named } => {
            assert_eq!(named, table.table_name());
        }
        other => panic!("expected NotRecomputable, got {other:?}"),
    }
}

/// The artifact digest a table's manifest attests — the byte-identity witness.
async fn artifact_digest(session: &InferenceSession, table: &str) -> String {
    let record = session
        .catalog()
        .get_result_table(table)
        .await
        .unwrap()
        .expect("table present");
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    session
        .result_store()
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("manifest sidecar present")
        .artifact
        .0
}

/// A materialised training set carries NO version, and the mechanism that
/// keeps it that way is a caller-side kind refusal in this crate — pinned here.
///
/// A read through a training set's session registration cannot straddle a
/// version boundary on a relation that never gets a second version. Every
/// verb that could publish one over a result table — `refresh_embeddings`
/// and `compact_embeddings`, plus `expire_versions`, which deletes versions
/// rather than publishing them — enters through
/// `InferenceSession::refreshable_pin`, which refuses any record whose
/// `kind` is not `ResultTableKind::Model` with
/// `NotRefreshable { NotEmbeddingTable }`. This test drives all three at a real
/// `ready` `TrainingSet` row and asserts the typed refusal plus the state it
/// leaves behind.
///
/// **This is caller discipline, not a storage-layer impossibility.**
/// Publishing a base version on a `TrainingSet` row through `Catalog::publish_base_version`
/// (`current_version` None → `Some(0)`) and allocating a second through
/// `ResultStore::allocate_version` (`Ok(1)`) both succeed. The `kind = 'model'` predicate in
/// `resolve_embedding_table` gates only source_id-addressed resolution, which none of these verbs
/// uses. Nothing in the schema keeps a version off a training-set row; only the
/// refusal asserted here does, so this oracle is the whole guard.
#[tokio::test(flavor = "multi_thread")]
async fn refresh_and_compaction_refuse_a_training_set_leaving_it_versionless() {
    use jammi_ai::pipeline::embedding_refresh::RefreshOptions;
    use jammi_db::error::{JammiError, NotRefreshableReason};

    let dir = TempDir::new().unwrap();
    let session = session_over(&dir, &common::fixture_url("training_pairs.csv")).await;
    let (table, _) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "training",
        &parity_columns(),
        ModelTask::TextEmbedding,
        "contrastive",
    )
    .await
    .unwrap();
    let name = table.table_name().to_string();

    // The preconditions the refusals have to be measured against: a REAL row,
    // `ready` (so `refreshable_pin`'s earlier `NotReady` arm cannot be what
    // answers), of kind `TrainingSet`, with no version yet.
    let before = session
        .catalog()
        .get_result_table(&name)
        .await
        .unwrap()
        .expect("the producer promoted a catalog row");
    assert_eq!(before.kind, ResultTableKind::TrainingSet);
    assert_eq!(
        before.status,
        ResultTableStatus::Ready.to_string(),
        "a non-ready row would be refused for an unrelated reason, making this \
         oracle vacuous about the KIND"
    );
    assert_eq!(common::current_version(&session, &name).await, None);

    // Each verb, driven at that row. `expire_versions` is asked for the widest
    // possible window so nothing but the refusal can be what stops it.
    let arms: Vec<(&str, std::result::Result<String, JammiError>)> = vec![
        (
            "refresh_embeddings",
            session
                .refresh_embeddings(&name, RefreshOptions::default())
                .await
                .map(|r| format!("{r:?}")),
        ),
        (
            "compact_embeddings",
            session
                .compact_embeddings(&name)
                .await
                .map(|r| format!("{r:?}")),
        ),
        (
            "expire_versions",
            session
                .expire_versions(&name, i64::MAX)
                .await
                .map(|r| format!("{r:?}")),
        ),
    ];
    for (verb, outcome) in arms {
        match outcome {
            Err(JammiError::NotRefreshable { table: t, reason }) => {
                assert_eq!(t, name, "{verb} must name the table it refused");
                assert_eq!(
                    reason,
                    NotRefreshableReason::NotEmbeddingTable,
                    "{verb} must refuse a TrainingSet as a non-embedding table"
                );
            }
            other => panic!(
                "{verb} must refuse a TrainingSet with \
                 NotRefreshable {{ NotEmbeddingTable }}, got {other:?}"
            ),
        }
    }

    // ... and the refusals left the relation versionless: no `current_version`,
    // no allocation consumed, no version row at all. The last is what the
    // allowlist entry actually needs — a read through the relation this row
    // registers has one and only one state to see.
    let after = session
        .catalog()
        .get_result_table(&name)
        .await
        .unwrap()
        .expect("a refusal never deletes the row");
    assert_eq!(
        common::pin(&session, after.clone()).await.version(),
        None,
        "a refused verb must publish no version"
    );
    assert_eq!(
        after.next_version, before.next_version,
        "a refused verb must not even consume a version number"
    );
    assert_eq!(after.status, ResultTableStatus::Ready.to_string());
    let versions = session
        .catalog()
        .list_result_table_versions(&name)
        .await
        .unwrap();
    assert!(
        versions.is_empty(),
        "a training set has no version rows, got {versions:?}"
    );
}
