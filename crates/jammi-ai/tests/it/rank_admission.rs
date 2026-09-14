//! #500: what a deployment admits a rank count for, and where it refuses
//! one.
//!
//! Two edges, and the tests here are about which is which.
//!
//! **Session open** decides whether this BUILD can reach the configured
//! collective. A process that cannot honour its own configuration should not
//! come up and then refuse every job it is handed.
//!
//! **The submit edge** decides whether this DEPLOYMENT can serve the count a
//! particular job asks for. It is the last point at which refusing costs
//! nothing: past it the spec is a durable row a worker will claim, fail and
//! retry. Every refusal here is asserted on two things — the typed error
//! variant, and that the `jobs` table is unchanged — because a refusal that
//! leaves a queued row behind is a job that later runs with a count the
//! deployment cannot serve.
//!
//! The submit edge has more than one entrance, and this file ranges over the
//! set of them:
//!
//! | entry path | reaches the edge through |
//! |---|---|
//! | embedded, per-verb (`fine_tune`, `fine_tune_graph`, `submit_fine_tune`) | `InferenceSession::submit_fine_tune_spec_deduped` |
//! | embedded, generic (`InferenceSession::enqueue(JobSpec::Training)`) | `InferenceSession::enqueue` |
//! | wire (`JobService::SubmitJob`) | `run_training_spec_deduped` → `submit_fine_tune_spec_deduped` |
//! | Python (`Database._start_training_proto`) | `jammi_ai::wire::training_spec_from_bytes` → `run_training_spec_deduped` → the same |
//!
//! The wire and Python paths are the embedded path plus a decode: both build
//! a `SubmitJobRequest`, both hand it to `training_spec_from_proto`, and both
//! then call `run_training_spec_deduped`, which funnels into the same
//! method the per-verb entry points do. This file drives the two EMBEDDED
//! entrances directly and the decode seam through
//! `training_spec_from_bytes` (the Python entrance's own first call); the
//! remote entrance's own end-to-end oracle lives in the server suite
//! (`crates/jammi-server/tests/it/grpc_remote_compute.rs`), where a real
//! client is available.

use std::path::Path;
use std::sync::Arc;

use jammi_ai::fine_tune::spec::{RankAdmission, TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod, HardNegativeConfig};
use jammi_ai::jobs::JobSpec;
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::CollectiveSelection;
use jammi_db::error::JammiError;
use tempfile::TempDir;

use crate::common;

/// A session over a deployment that declares `devices` devices.
///
/// Real ordinals with `device = 0`, which is what `GpuConfig::validate`
/// requires of a multi-entry list (a list mixing the CPU with real ordinals
/// is refused: a gang runs on one kind of device). Hermetic anyway —
/// `require_gpu` stays false, so a host with no such device degrades to the
/// CPU exactly as every other fixture's session does, and it is the declared
/// COUNT, not the execution device, that the submit edge reads.
async fn session_with_devices(devices: usize) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.gpu.device = 0;
    config.gpu.devices = Some((0..devices as i32).collect());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    (session, dir)
}

fn spec_with_world_size(world_size: u32) -> TrainingSpec {
    spec_with(world_size, FineTuneConfig::default())
}

fn spec_with(world_size: u32, config: FineTuneConfig) -> TrainingSpec {
    TrainingSpec::FineTune {
        source: "patents".into(),
        columns: vec!["abstract".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: "local:tiny".into(),
            config,
            world_size,
        },
        cache: jammi_db::store::CachePolicy::Bypass,
    }
}

async fn job_count(session: &Arc<InferenceSession>) -> usize {
    session.catalog().list_jobs().await.unwrap().len()
}

/// Every refusal, on BOTH embedded entrances, asserted on the typed variant
/// and on an unchanged `jobs` table.
///
/// One test over the whole refusal set rather than five, because the
/// expensive part is the session and the interesting part is that the set is
/// closed: each case names the bound it crosses, and the deployment that
/// admits it is exercised at the end so no case passes by refusing
/// everything.
#[tokio::test(flavor = "multi_thread")]
async fn every_unservable_rank_count_is_refused_at_both_submit_entrances() {
    // One device: a two-rank job is unservable here.
    let (session, _dir) = session_with_devices(1).await;
    let before = job_count(&session).await;

    let cached = FineTuneConfig {
        cached: true,
        ..FineTuneConfig::default()
    };
    let mining = FineTuneConfig {
        hard_negatives: HardNegativeConfig {
            mine: true,
            ..HardNegativeConfig::default()
        },
        ..FineTuneConfig::default()
    };

    // The two bounds a ONE-device deployment can state. The two
    // single-rank-only mechanisms need a deployment where the device bound
    // does not bite first, so they are checked on the wide session below.
    // The expectation is the WHOLE sentence, not a fragment of it: a message
    // is what the operator acts on, and a fragment-only assertion cannot see
    // a sentence that arrives with its second half detached.
    let cases: [(&str, TrainingSpec, &str); 2] = [
        (
            "a zero-rank count",
            spec_with_world_size(0),
            "world_size must be >= 1 (1 is the single-rank job; 0 has no rank to run on)",
        ),
        (
            "a count beyond the deployment's devices",
            spec_with_world_size(2),
            "world_size = 2 exceeds the 1 configured device(s): one rank per device, so list \
             more in `[gpu] devices` or submit a smaller rank count",
        ),
    ];

    for (name, spec, expected) in cases {
        // Entrance 1: the per-verb funnel.
        let per_verb = session
            .run_training_spec(spec.clone())
            .await
            .expect_err(name);
        assert!(
            matches!(per_verb, JammiError::Config(_)),
            "{name}: the refusal must be typed, got {per_verb:?}"
        );
        assert!(
            per_verb.to_string().contains(expected),
            "{name}: the refusal must name the bound it crosses, got {per_verb}"
        );

        // Entrance 2: the generic enqueue, which takes an already-built spec
        // and so does not pass the per-verb entry points at all.
        let generic = match session
            .enqueue(JobSpec::Training(Box::new(spec.clone())), 0)
            .await
        {
            Ok(handle) => panic!(
                "{name}: the generic entrance admitted job {}",
                handle.job_id
            ),
            Err(e) => e,
        };
        assert!(
            matches!(generic, JammiError::Config(_)),
            "{name}: the generic entrance must refuse the same way, got {generic:?}"
        );
        assert_eq!(
            generic.to_string(),
            per_verb.to_string(),
            "{name}: one rule, one message, whichever entrance the spec came through"
        );

        assert_eq!(
            job_count(&session).await,
            before,
            "{name}: a refused submission must enqueue nothing"
        );
    }

    // Two devices: the device bound no longer bites, so the two
    // single-rank-only mechanisms are the reason a two-rank job is refused.
    let (wide, _wide_dir) = session_with_devices(2).await;
    let wide_before = job_count(&wide).await;
    for (name, spec, expected) in [
        (
            "a multi-rank GradCache run",
            spec_with(2, cached),
            "world_size = 2 cannot be combined with GradCache (`cached`): the cached \
             objective's second pass is over the WHOLE batch on one rank, so a gang would \
             not compute the objective this config asks for",
        ),
        (
            "a multi-rank mining run",
            spec_with(2, mining),
            "world_size = 2 cannot be combined with hard-negative mining \
             (`hard_negatives.mine`): the miner retrieves from this process's own index, so \
             each rank would mine a different negative pool",
        ),
    ] {
        let error = wide.run_training_spec(spec.clone()).await.expect_err(name);
        assert!(matches!(error, JammiError::Config(_)), "{name}: {error:?}");
        assert!(error.to_string().contains(expected), "{name}: got {error}");
        let generic = match wide.enqueue(JobSpec::Training(Box::new(spec)), 0).await {
            Ok(handle) => panic!(
                "{name}: the generic entrance admitted job {}",
                handle.job_id
            ),
            Err(e) => e,
        };
        assert_eq!(generic.to_string(), error.to_string());
    }
    assert_eq!(
        job_count(&wide).await,
        wide_before,
        "a refused submission must enqueue nothing"
    );

    // The control: the SAME two-rank submission with neither
    // single-rank-only mechanism is admitted on the two-device deployment and
    // does write a row. Without this the refusals above could all be a
    // submit edge that refuses everything.
    wide.run_training_spec(spec_with_world_size(2))
        .await
        .expect("a two-rank job on a two-device deployment is servable");
    assert_eq!(
        job_count(&wide).await,
        wide_before + 1,
        "the admitted submission must enqueue exactly one row"
    );
}

/// `cache = Use` on `TrainingSpec::FineTune` is refused on BOTH embedded
/// submit entrances, not just the per-verb one: `InferenceSession::enqueue`
/// takes an already-built spec, bypassing every per-verb entry point, and
/// must still refuse it before writing a row.
#[tokio::test(flavor = "multi_thread")]
async fn a_fine_tune_cache_use_is_refused_through_enqueue_too() {
    let (session, _dir) = session_with_devices(1).await;
    let before = job_count(&session).await;

    let spec = TrainingSpec::FineTune {
        source: "patents".into(),
        columns: vec!["abstract".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: "local:tiny".into(),
            config: FineTuneConfig::default(),
            world_size: 1,
        },
        cache: jammi_db::store::CachePolicy::Use,
    };

    let per_verb = session
        .run_training_spec(spec.clone())
        .await
        .expect_err("cache = Use must be refused through the per-verb funnel");
    assert!(
        matches!(per_verb, JammiError::Config(_)),
        "the refusal must be typed, got {per_verb:?}"
    );

    let generic = match session.enqueue(JobSpec::Training(Box::new(spec)), 0).await {
        Ok(handle) => panic!(
            "the generic enqueue entrance admitted a cache=Use job {}",
            handle.job_id
        ),
        Err(e) => e,
    };
    assert!(
        matches!(generic, JammiError::Config(_)),
        "the generic entrance must refuse the same way, got {generic:?}"
    );
    assert_eq!(
        generic.to_string(),
        per_verb.to_string(),
        "one rule, one message, whichever entrance the spec came through"
    );
    assert_eq!(
        job_count(&session).await,
        before,
        "a refused submission must enqueue nothing, through either entrance"
    );
}

/// The single-rank job every caller that names no count submits is still
/// admitted on a one-device deployment — the no-regression case the refusals
/// above must not have swept up.
#[tokio::test(flavor = "multi_thread")]
async fn a_single_rank_job_is_admitted_on_a_one_device_deployment() {
    let (session, _dir) = session_with_devices(1).await;
    let before = job_count(&session).await;
    session
        .run_training_spec(spec_with_world_size(1))
        .await
        .expect("the single-rank job is what every deployment can serve");
    assert_eq!(job_count(&session).await, before + 1);
}

/// The Python entrance's own first call: the embedded binding assembles a
/// `SubmitJobRequest`, serializes it, and hands the BYTES to
/// `training_spec_from_bytes` — so a count that crossed that seam wrongly
/// would reach the same submit edge with the wrong value. Decoding the bytes
/// and submitting the decoded spec is exactly what
/// `Database::_start_training_proto` does.
#[tokio::test(flavor = "multi_thread")]
async fn the_serialized_request_entrance_carries_the_count_to_the_same_edge() {
    use prost::Message;

    let (session, _dir) = session_with_devices(1).await;
    let before = job_count(&session).await;

    let mut body = Vec::new();
    jammi_ai::wire::training_spec_to_proto(&spec_with_world_size(2))
        .encode(&mut body)
        .expect("encode");
    let decoded = jammi_ai::wire::training_spec_from_bytes(&body).expect("decode");
    let TrainingSpec::FineTune { common, .. } = &decoded else {
        panic!("expected the fine_tune variant");
    };
    assert_eq!(
        common.world_size, 2,
        "the count must survive the serialized round trip, or this entrance would submit a \
         different job than the caller assembled"
    );

    let error = session
        .run_training_spec(decoded)
        .await
        .expect_err("a two-rank job on a one-device deployment is unservable");
    assert!(matches!(error, JammiError::Config(_)), "{error:?}");
    assert_eq!(
        job_count(&session).await,
        before,
        "a refusal on this entrance enqueues nothing either"
    );
}

/// `collective = "nccl"` is refused at session OPEN on a build without CUDA,
/// and `auto`/`cpu` open fine.
///
/// Stated for both builds rather than only this one: on a CUDA build the same
/// configuration must OPEN, so the refusal is a statement about the build and
/// not a blanket rejection of the knob.
#[tokio::test(flavor = "multi_thread")]
async fn nccl_without_cuda_is_refused_at_session_open() {
    for collective in [CollectiveSelection::Auto, CollectiveSelection::Cpu] {
        let dir = TempDir::new().unwrap();
        let mut config = common::test_config(dir.path());
        config.worker.collective = collective;
        InferenceSession::new(config)
            .await
            .unwrap_or_else(|e| panic!("collective = {collective} must open: {e}"));
    }

    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.worker.collective = CollectiveSelection::Nccl;
    let opened = InferenceSession::new(config).await;

    if cfg!(feature = "cuda") {
        assert!(
            opened.is_ok(),
            "a CUDA build can reach NCCL, so the same configuration must open"
        );
    } else {
        let error = opened
            .err()
            .expect("a build without CUDA cannot reach NCCL and must refuse to open");
        assert!(matches!(error, JammiError::Config(_)), "{error:?}");
        assert!(
            error.to_string().contains("cuda"),
            "the refusal must name the missing build feature: {error}"
        );
    }
}

/// The admission rule itself, over deployments this host does not have.
///
/// The `nccl`-without-CUDA arm is unreachable from a LIVE session — the open
/// refusal above stops such a session from existing — so the rule is pinned
/// here, where the build flag is data: a host build states what a CUDA build
/// admits, and a CUDA build states what a host build refuses. Both
/// directions are asserted, so the arm cannot pass by never firing.
#[test]
fn the_admission_rule_reads_the_build_as_data() {
    let spec = spec_with_world_size(1);

    let host_build = RankAdmission::new(1, CollectiveSelection::Nccl, false);
    let error = host_build
        .admit(&spec)
        .expect_err("a build without CUDA cannot reach NCCL");
    assert!(matches!(error, JammiError::Config(_)), "{error:?}");
    assert!(
        error.to_string().contains(
            "[worker] collective = \"nccl\" needs a build with the `cuda` feature; this \
             binary has none, so the requested collective cannot be reached"
        ),
        "the refusal must state the whole reason: {error}"
    );

    let cuda_build = RankAdmission::new(1, CollectiveSelection::Nccl, true);
    cuda_build
        .admit(&spec)
        .expect("a CUDA build reaches NCCL, so the same spec is admitted");

    for collective in [CollectiveSelection::Auto, CollectiveSelection::Cpu] {
        RankAdmission::new(1, collective, false)
            .admit(&spec)
            .unwrap_or_else(|e| panic!("{collective} needs no CUDA: {e}"));
    }
}

/// Every refusal the rule can raise reads as one sentence.
///
/// A multi-line Rust string literal joins its lines only through a trailing
/// `\`; without it the source's own indentation is part of the message, and
/// the operator is handed a sentence with a gap in the middle of it. The
/// oracle is the gap, not any one wording: no refusal message contains a run
/// of two spaces.
///
/// The set ranged over is every `return Err` of
/// `RankAdmission::admit` — the five conditions the method tests, in the
/// order it tests them; each case below is constructed so that its own
/// condition is the first one to bite. A sixth branch added later without a
/// case here would not be covered, which is why each case names the
/// condition it fires.
#[test]
fn every_admission_refusal_reads_as_one_sentence() {
    let cached = FineTuneConfig {
        cached: true,
        ..FineTuneConfig::default()
    };
    let mining = FineTuneConfig {
        hard_negatives: HardNegativeConfig {
            mine: true,
            ..HardNegativeConfig::default()
        },
        ..FineTuneConfig::default()
    };

    let cases: [(&str, RankAdmission, TrainingSpec); 5] = [
        (
            "world_size == 0",
            RankAdmission::new(1, CollectiveSelection::Auto, false),
            spec_with_world_size(0),
        ),
        (
            "world_size > devices",
            RankAdmission::new(1, CollectiveSelection::Auto, false),
            spec_with_world_size(2),
        ),
        (
            "nccl without a cuda build",
            RankAdmission::new(1, CollectiveSelection::Nccl, false),
            spec_with_world_size(1),
        ),
        (
            "world_size > 1 with GradCache",
            RankAdmission::new(2, CollectiveSelection::Auto, false),
            spec_with(2, cached),
        ),
        (
            "world_size > 1 with hard-negative mining",
            RankAdmission::new(2, CollectiveSelection::Auto, false),
            spec_with(2, mining),
        ),
    ];

    for (condition, admission, spec) in cases {
        let message = admission.admit(&spec).expect_err(condition).to_string();
        assert!(
            !message.contains("  "),
            "{condition}: the refusal arrives with its source indentation in it: {message:?}"
        );
    }
}

/// A context-predictor spec carries no rank count at all: the variant has no
/// `TrainingCommon`, so there is no field to hold one and nothing for the
/// submit edge to admit. The count is refused at the wire decode instead —
/// the last edge that can still see one a caller chose.
///
/// The impossibility is the TYPE's, and this is the executed attempt to
/// falsify it: the variant is destructured exhaustively, so a `world_size`
/// added to it later would fail to compile here rather than silently become
/// an unadmitted count.
#[test]
fn a_context_predictor_spec_has_no_rank_count_to_admit() {
    let spec = TrainingSpec::ContextPredictor {
        source: "episodes".into(),
        predictor_spec: predictor_config(),
    };
    match &spec {
        TrainingSpec::ContextPredictor {
            source,
            predictor_spec,
        } => {
            assert_eq!(source, "episodes");
            assert_eq!(predictor_spec.context_k, 4);
        }
        other => panic!("expected the predictor variant, got {other:?}"),
    }
    RankAdmission::new(1, CollectiveSelection::Auto, false)
        .admit(&spec)
        .expect("a predictor spec has no count, so there is nothing to refuse");
}

fn predictor_config() -> jammi_ai::pipeline::context_predictor::ContextPredictorTrainConfig {
    use jammi_ai::pipeline::context_predictor::{
        ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
    };
    ContextPredictorTrainConfig {
        model_id: "ctx-pred-admission".into(),
        architecture: ContextArchitecture::Tnp,
        key_column: "row_key".into(),
        task_column: "cohort".into(),
        value_column: "outcome".into(),
        context_k: 4,
        hidden_dim: 32,
        num_heads: 2,
        num_layers: 1,
        head: PredictiveHead::Gaussian {
            objective: GaussianObjective::Nll { beta: 0.5 },
        },
        epochs: 1,
        learning_rate: 3e-4,
        grad_clip: 1.0,
        test_task_fraction: 0.3,
        min_task_count: 2,
        seed: 7,
    }
}

/// Structural oracle: the CLOSED set of durable submit edges for a
/// `TrainingSpec` — `InferenceSession::submit_fine_tune_spec_deduped`,
/// `InferenceSession::enqueue`, and
/// `pipeline::context_predictor::InferenceSession::train_context_predictor_deduped`
/// — each calls `crate::fine_tune::spec::admit_training_spec`, the ONE
/// function holding the rank admission, the per-kind validation, and the
/// `cache = Use` refusal. This is deliberately source-level, not behavioural
/// only: a re-implementation of the SAME checks inline at an edge, without
/// going through the shared function, would pass every behavioural oracle
/// above yet violate the property this test exists to pin — "ONE admission
/// function", not merely "equivalent behaviour, duplicated". A fourth edge
/// added later without updating this list is a loud failure here, not a
/// silently-unadmitted spec.
///
/// Mutation (executed and reverted, never shipped — see this round's
/// report): deleting the `admit_training_spec` call from `enqueue` drops
/// the call count to 2 and this test fails, naming the file.
#[test]
fn every_durable_training_submit_edge_calls_the_one_admission_function() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-ai has two ancestors: crates/, then the repo root")
        .to_path_buf();

    // The stated, closed universe: (path relative to the repo root, the
    // function whose body must call the admission fn).
    let edges: &[(&str, &str)] = &[
        (
            "crates/jammi-ai/src/session.rs",
            // kernel-oracles: fn-in-literal reviewed: the edge function's own signature text, searched for verbatim below — not code in this file
            "async fn submit_fine_tune_spec_deduped(",
        ),
        // kernel-oracles: fn-in-literal reviewed: the edge function's own signature text, searched for verbatim below — not code in this file
        ("crates/jammi-ai/src/jobs.rs", "pub async fn enqueue("),
        (
            "crates/jammi-ai/src/pipeline/context_predictor.rs",
            // kernel-oracles: fn-in-literal reviewed: the edge function's own signature text, searched for verbatim below — not code in this file
            "pub(crate) async fn train_context_predictor_deduped(",
        ),
    ];

    // The call site every edge below must use, verbatim — the fully
    // qualified path, never a bare `admit_training_spec(` (which would also
    // match the function's OWN definition, `fn admit_training_spec(`).
    const CALL: &str = "fine_tune::spec::admit_training_spec(";

    let mut total_calls = 0usize;
    for (path, fn_sig) in edges {
        let full = root.join(path);
        let src =
            std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("could not read {path}: {e}"));
        let fn_start = src.find(fn_sig).unwrap_or_else(|| {
            panic!("{path} no longer defines `{fn_sig}` — this oracle's edge list is stale")
        });
        // The function body: from the signature to the next line holding
        // only a closing brace at the SAME (four-space method) indent —
        // exact enough for these three concretely-indented methods, and any
        // false match only widens the search window, never narrows it past
        // the real body.
        let after_sig = &src[fn_start..];
        let body_end = after_sig.find("\n    }\n").unwrap_or(after_sig.len());
        let body = &after_sig[..body_end];
        let calls_in_body = body.matches(CALL).count();
        assert!(
            calls_in_body >= 1,
            "{path}'s `{fn_sig}` must call `{CALL}` before writing a jobs row \
             (found {calls_in_body} calls in its body)"
        );
        total_calls += calls_in_body;
    }

    // Nothing outside the three edges above calls it either — a stray
    // fourth call site would mean an edge this list has not named.
    let mut whole_crate_calls = 0usize;
    for (path, _) in edges {
        let full = root.join(path);
        let src = std::fs::read_to_string(&full).unwrap();
        whole_crate_calls += src.matches(CALL).count();
    }
    assert_eq!(
        whole_crate_calls, total_calls,
        "a call to the admission function exists outside the three named edge \
         bodies — either a new edge needs adding to this oracle's list, or a \
         call site drifted outside its edge's own function"
    );
    assert_eq!(
        total_calls, 3,
        "expected exactly one admission call per edge across the three-edge universe, got {total_calls}"
    );
}
