//! The SDK front door, embedded path. `Jammi::open(Target::Local(config))` must
//! yield a working in-process [`Session`] — the one-call "use the SDK, run any
//! shape" entry point. This drives the real source → generate-embeddings →
//! search pipeline over the patents fixture and the tiny BERT cookbook model
//! through the `Session` the factory returns, proving the front door produces a
//! live embedded session, not just a constructed value.

use arrow::array::StringArray;
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod, LrSchedule};
use jammi_ai::local_session::FineTuneJobId;
use jammi_ai::model::ModelTask;
use jammi_ai::{Jammi, Modality, SearchQuery, SearchRequest, Session, Target};
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

fn tiny_bert() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// `Jammi::open(Target::Local(_))` returns an embedded [`Session`] that drives
/// the full embed → search pipeline end to end against the patents corpus.
#[tokio::test]
async fn open_local_yields_a_working_embedded_session() {
    let dir = TempDir::new().unwrap();
    let session: Session = Jammi::open(Target::Local(common::test_config(dir.path())))
        .await
        .expect("open local session");

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
        .expect("add_source through the opened session");

    let record = session
        .generate_embeddings(
            "patents",
            &tiny_bert(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect("generate_embeddings through the opened session")
        .0;
    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0, "patents corpus embeds rows");

    let hits = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::Vector(vec![0.5_f32; 32]),
            k: 5,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            oversample: None,
        })
        .await
        .expect("search through the opened session");
    assert!(!hits.is_empty(), "the opened session returns search hits");
    assert!(
        hits.iter().any(|b| b
            .column_by_name("_row_id")
            .is_some_and(|c| { c.as_any().downcast_ref::<StringArray>().is_some() })),
        "hits carry the _row_id provenance column the search verb yields"
    );
}

// ---------------------------------------------------------------------------
// GAP-A-5 (#446): `[training] run_worker` on the RUST SDK arm.
//
// The design's one-binary symmetry claim (B4/K4) is that the server, the Python
// embedded binding, and the Rust SDK front door all decide "does THIS process
// claim training jobs?" by reading the SAME configuration key — not by three
// private conventions. The server arm is proven in
// `crates/jammi-server/tests/it/grpc_training.rs`
// (`train_tier_with_run_worker_{false,true}_*`) and the Python arm in
// `crates/jammi-python`; these are the Rust SDK arm's peers, driven through the
// SAME `jammi.toml` -> `JammiConfig::load` path a real embedding binary takes,
// never through a struct literal (a literal would prove only that a field can
// be set, not that the deployed configuration reaches the spawn decision).
// ---------------------------------------------------------------------------

/// Build the front door's config the way a real embedded binary does: a
/// `jammi.toml` written into `dir` and read back through `JammiConfig::load`,
/// with `[training] run_worker` either set to `run_worker` or (when `None`) the
/// section omitted entirely so the default direction is what is under test.
///
/// `expected` is asserted on the LOADED config before it is handed to the front
/// door: `JammiConfig::load` applies `JAMMI_*` environment overrides, so an
/// ambient `JAMMI_TRAINING__RUN_WORKER` in the test runner's environment would
/// otherwise silently invert the oracle. It fails loud here instead.
///
/// The rest mirrors `jammi_test_utils::test_config` (CPU device, small batch,
/// temp artifact dir) so the engine under test is the one every other
/// integration test opens.
fn front_door_config(
    dir: &std::path::Path,
    run_worker: Option<bool>,
    expected: bool,
) -> JammiConfig {
    let training = match run_worker {
        Some(v) => format!("\n[training]\nrun_worker = {v}\n"),
        None => String::new(),
    };
    let config_path = dir.join("jammi.toml");
    std::fs::write(
        &config_path,
        format!(
            "artifact_dir = \"{artifact_dir}\"\n\
             \n\
             [gpu]\n\
             device = -1\n\
             \n\
             [inference]\n\
             batch_size = 8\n\
             \n\
             [logging]\n\
             level = \"debug\"\n\
             {training}",
            artifact_dir = dir.display(),
        ),
    )
    .expect("write the fixture's jammi.toml");

    let config = JammiConfig::load(Some(&config_path)).expect("the fixture's jammi.toml loads");
    assert_eq!(
        config.training.run_worker, expected,
        "the loaded config must carry the run_worker this fixture asked for — a mismatch \
         means an ambient JAMMI_TRAINING__RUN_WORKER override is inverting the oracle"
    );
    assert_eq!(
        config.artifact_dir.as_path(),
        dir,
        "the loaded config must root its artifacts in this fixture's temp dir"
    );
    config
}

/// Register the training-pairs fixture as the source the submissions below
/// fine-tune over.
async fn seed_training_source(session: &Session) {
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .expect("add the training source through the opened session");
}

fn training_columns() -> Vec<String> {
    vec![
        "text_a".to_string(),
        "text_b".to_string(),
        "score".to_string(),
    ]
}

/// The cheapest real fine-tune shape: one epoch of the projection-head arm
/// (`target_modules` empty), which is enough for a job to be claimed and to
/// leave `queued` — the only property these tests read off it.
fn cheap_fine_tune_config() -> FineTuneConfig {
    FineTuneConfig {
        epochs: 1,
        batch_size: 4,
        warmup_steps: 0,
        lr_schedule: LrSchedule::Constant,
        ..Default::default()
    }
}

/// Submit a fine-tune job through the front-door session and return its id.
async fn submit(session: &Session) -> FineTuneJobId {
    session
        .fine_tune(
            "training",
            &tiny_bert(),
            &training_columns(),
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(cheap_fine_tune_config()),
        )
        .await
        .expect(
            "fine_tune must submit regardless of run_worker — the key gates CLAIMING, \
             never the submission surface",
        )
}

/// THE BINDING ORACLE for the Rust SDK arm: a front-door session opened from a
/// `jammi.toml` carrying `run_worker = false` accepts a `fine_tune` submission
/// and then never claims it — the job's `queued` status and its byte-exact
/// `{"state":"pending"}` acceleration marker are STABLE, not merely observed
/// once.
///
/// Stability is what makes this an oracle rather than a lucky read. The polls
/// span strictly MORE than the config's own `idle_poll_secs` (the interval at
/// which a claim loop, if one existed in this process, would tick and claim a
/// `queued` row — its first tick has no initial sleep at all), so a spawned
/// worker could not hide inside the observation window. Both fields are
/// re-asserted at EVERY poll, so a claim landing at any point in the span fails
/// the test. Against the pre-fix `Jammi::open` (which spawned unconditionally)
/// this fails.
///
/// The control that this is caused by the KNOB and not by a front door that
/// could never run anything is
/// `front_door_with_the_default_config_claims_the_submitted_job`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn front_door_with_run_worker_false_leaves_the_job_queued_and_pending_stable() {
    let dir = TempDir::new().unwrap();
    let session = Jammi::open(Target::Local(front_door_config(
        dir.path(),
        Some(false),
        false,
    )))
    .await
    .expect("open local session with run_worker = false");
    seed_training_source(&session).await;

    let job_id = submit(&session).await;
    assert!(!job_id.0.is_empty(), "submission returns a job id");

    // Span strictly more than one idle poll interval, read off the session's OWN
    // config so the window follows the knob's timing rather than a magic number.
    let idle_poll = std::time::Duration::from_secs(
        session
            .engine()
            .inner_config()
            .training
            .idle_poll_secs
            .max(1),
    );
    const POLLS: u32 = 6;
    let step = (idle_poll * 2) / POLLS;
    let began = std::time::Instant::now();

    for poll in 0..POLLS {
        let status = session
            .fine_tune_status(&job_id)
            .await
            .expect("fine_tune_status is served regardless of run_worker");
        assert_eq!(
            status, "queued",
            "poll {poll}: with run_worker = false NOTHING in this process may claim the \
             job — it must still read `queued`"
        );
        let record = session
            .engine()
            .catalog()
            .get_training_job(&job_id.0)
            .await
            .expect("get_training_job");
        assert_eq!(
            record.claimed_by, None,
            "poll {poll}: an unclaimed job carries no claimant — a `claimed_by` here means \
             a claim loop ran despite run_worker = false"
        );
        assert_eq!(
            record.acceleration_report.as_deref(),
            Some(r#"{"state":"pending"}"#),
            "poll {poll}: the submission-time acceleration marker must be STABLE \
             byte-for-byte under run_worker = false"
        );
        if poll + 1 < POLLS {
            tokio::time::sleep(step).await;
        }
    }

    assert!(
        began.elapsed() > idle_poll,
        "the observation window ({:?}) must exceed one idle_poll_secs ({idle_poll:?}) — \
         otherwise a claim loop could have been hiding between the polls",
        began.elapsed()
    );
}

/// THE CONTROL for the oracle above, and the pin on the DEFAULT direction: the
/// same front door, the same submission, from a `jammi.toml` with no
/// `[training]` section at all. The job LEAVES `queued`.
///
/// Without this, the stable-`queued` assertion above would pass just as well
/// against a front door that could never run anything (a broken worker, a
/// wedged catalog), so this is what makes the knob the demonstrated cause. It
/// also pins that an unconfigured embedded SDK consumer still gets a whole
/// engine — one that both submits and runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn front_door_with_the_default_config_claims_the_submitted_job() {
    let dir = TempDir::new().unwrap();
    let session = Jammi::open(Target::Local(front_door_config(dir.path(), None, true)))
        .await
        .expect("open local session with the default config");
    seed_training_source(&session).await;

    let job_id = submit(&session).await;

    // Bounded well past the window the run_worker = false oracle held the job
    // `queued` across, so "left queued" here is a real difference in behaviour
    // and not a difference in patience.
    let mut left_queued = None;
    for _ in 0..600 {
        let status = session.fine_tune_status(&job_id).await.expect("status");
        if status != "queued" {
            left_queued = Some(status);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let observed = left_queued.expect(
        "CONTROL: with the default config the front door's worker must claim the submitted \
         job — it never left `queued`, which would make the run_worker = false oracle vacuous",
    );
    assert_ne!(observed, "queued", "the claimed job has left the queue");
}

/// Dropping a `run_worker = false` front-door session must not panic or hang on
/// a worker that was never started, with a job still outstanding — the state
/// such a process is normally in.
///
/// The absence is carried by the session's `Option` worker slot being `None`,
/// so `Drop` has nothing to signal or abort. The bound is what makes this an
/// assertion: a drop path that awaited a never-spawned worker would sit here
/// until the timeout rather than returning.
///
/// Honest about its own power: this is a TEARDOWN guard on the `None` path, not
/// a second oracle for the knob. It passes against the pre-fix front door too
/// (`EmbeddedWorker::drop` signals and aborts without blocking, so a session
/// that DID spawn also drops promptly) — the knob itself is proven by
/// `front_door_with_run_worker_false_leaves_the_job_queued_and_pending_stable`,
/// which fails there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_run_worker_false_session_with_a_job_outstanding_is_immediate() {
    let dir = TempDir::new().unwrap();
    let session = Jammi::open(Target::Local(front_door_config(
        dir.path(),
        Some(false),
        false,
    )))
    .await
    .expect("open local session with run_worker = false");
    seed_training_source(&session).await;

    let job_id = submit(&session).await;
    assert_eq!(
        session.fine_tune_status(&job_id).await.unwrap(),
        "queued",
        "the job is outstanding at drop — that is the state this teardown must handle"
    );

    let began = std::time::Instant::now();
    drop(session);
    assert!(
        began.elapsed() < std::time::Duration::from_secs(5),
        "dropping a session that never spawned a worker must return immediately, took {:?}",
        began.elapsed()
    );
}
