//! What a deployment admits a rank count for, and where it refuses
//! one.
//!
//! Two edges, and the tests here are about which is which.
//!
//! **Session open** decides whether this BUILD can reach the configured
//! collective. A process that cannot honour its own configuration should not
//! come up and then refuse every job it is handed.
//!
//! **The submit edge** decides whether this DEPLOYMENT can serve the count a
//! particular job asks for — the FLEET's bound, `[distributed]
//! max_world_size` (the serveable world), never this host's own device
//! count: a count within the serveable world but beyond this host's devices
//! submits and is decided by assembly on the claiming coordinator. It is the last point at which refusing costs nothing: past it
//! the spec is a durable row a worker will claim, fail and retry. Every
//! refusal here is asserted on two things — the typed error variant, and
//! that the `jobs` table is unchanged — because a refusal that leaves a
//! queued row behind is a job that later runs with a count the deployment
//! cannot serve. The rule reads no catalog: `RankAdmission` holds no handle
//! to one, so the refusal is decided from configuration alone.
//!
//! The submit edge has more than one entrance, and this file ranges over the
//! set of them:
//!
//! | entry path | reaches the edge through |
//! |---|---|
//! | embedded, per-verb (`fine_tune`, `fine_tune_graph`, `submit_fine_tune`) | `InferenceSession::submit_fine_tune_spec_deduped` |
//! | embedded, generic (`InferenceSession::enqueue(JobSpec::FineTune { .. })`) | `InferenceSession::enqueue` |
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

/// A session over a deployment whose serveable world is `world`
/// (`[distributed] max_world_size`), on ONE device (the fixture's CPU): the
/// submit edge reads the fleet bound, never the device count, and every
/// refusal below is a statement about that bound.
async fn session_with_serveable_world(world: u32) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.distributed.max_world_size = world;
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
    // A serveable world of one: a two-rank job is unservable here.
    let (session, _dir) = session_with_serveable_world(1).await;
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

    // The two bounds a serveable-world-of-one deployment can state. The two
    // single-rank-only mechanisms need a deployment where the world bound
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
            "a count beyond the deployment's serveable world",
            spec_with_world_size(2),
            "world_size = 2 exceeds the serveable world of 1 ([distributed] max_world_size): a \
             gang is assembled from fleet members up to that bound, so raise it on every \
             coordinator or submit a smaller rank count",
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
        let generic = match session.enqueue(JobSpec::from(spec.clone()), 0).await {
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

    // A serveable world of two: the world bound no longer bites, so the two
    // single-rank-only mechanisms are the reason a two-rank job is refused.
    let (wide, _wide_dir) = session_with_serveable_world(2).await;
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
        let generic = match wide.enqueue(JobSpec::from(spec), 0).await {
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
    // single-rank-only mechanism is admitted on the serveable-world-of-two
    // deployment and does write a row — on ONE device: a count within the
    // serveable world but beyond this host's own devices submits and is decided by assembly (the coordinator body's
    // own oracle, `gang_coordinator.rs`), never refused here. Without this
    // the refusals above could all be a submit edge that refuses everything.
    assert_eq!(
        wide.inner_config().gpu.device_list().len(),
        1,
        "the fixture declares one device, so the admitted count is beyond this host's devices"
    );
    wide.run_training_spec(spec_with_world_size(2))
        .await
        .expect("a two-rank job within the serveable world submits on a one-device host");
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
    let (session, _dir) = session_with_serveable_world(1).await;
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

    let generic = match session.enqueue(JobSpec::from(spec), 0).await {
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
/// admitted on a serveable world of one — the no-regression case the
/// refusals above must not have swept up.
#[tokio::test(flavor = "multi_thread")]
async fn a_single_rank_job_is_admitted_on_a_serveable_world_of_one() {
    let (session, _dir) = session_with_serveable_world(1).await;
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

    let (session, _dir) = session_with_serveable_world(1).await;
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
        .expect_err("a two-rank job on a serveable world of one is unservable");
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

/// The refusing half, at the rule itself: a
/// `world_size` past the serveable world is refused naming `[distributed]
/// max_world_size`, and one within it is admitted — decided with NO
/// catalog in scope at all. `RankAdmission` is three plain values (the
/// serveable world, the collective, the build flag): it holds no catalog
/// handle, so no catalog read is even expressible from `admit` — the
/// structural half of "refuses at submit with no catalog read"; the
/// session-level half (the `jobs` table unchanged) is
/// `every_unservable_rank_count_is_refused_at_both_submit_entrances`.
#[test]
fn a_count_past_the_serveable_world_is_refused_from_configuration_alone() {
    let narrow = RankAdmission::new(1, CollectiveSelection::Auto, false);
    let error = narrow
        .admit(&spec_with_world_size(2))
        .expect_err("two ranks on a serveable world of one");
    assert!(matches!(error, JammiError::Config(_)), "{error:?}");
    assert!(
        error.to_string().contains("[distributed] max_world_size"),
        "the refusal names the knob that bounds it: {error}"
    );
    assert_eq!(narrow.serveable_world(), 1);

    let wide = RankAdmission::new(2, CollectiveSelection::Auto, false);
    wide.admit(&spec_with_world_size(2))
        .expect("two ranks within a serveable world of two are admitted");
    wide.admit(&spec_with_world_size(3))
        .expect_err("three ranks past a serveable world of two are refused");
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
            "world_size > serveable_world",
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

/// The context-predictor behavioural oracle: the edge's only admission
/// effect for this kind is `ContextPredictorTrainConfig::validate` (the
/// spec above shows `RankAdmission::admit` is a no-op for it), so this is
/// the ONE test that can go red if that validation is ever skipped —
/// submit an invalid predictor config through the real durable edge
/// (`InferenceSession::train_context_predictor`, which calls
/// `train_context_predictor_deduped`) and assert a typed refusal with
/// nothing enqueued. RED (executed and reverted): replacing
/// `train_context_predictor_deduped`'s `let admitted = admit_training_spec(
/// ..., training_spec)?; let training_spec = admitted.spec();` with `let
/// training_spec = &training_spec;` (bypassing admission and borrowing the
/// spec directly — compiles fine, since nothing here forces every edge to
/// use the witness) turns this AND
/// `every_durable_training_submit_edge_calls_the_one_admission_function`
/// red: this test fails with an untyped `Catalog` error from further
/// downstream (validation never ran) and the source oracle reports 0 calls
/// in the edge's body.
#[tokio::test(flavor = "multi_thread")]
async fn an_invalid_context_predictor_config_is_refused_through_the_real_edge() {
    let (session, _dir) = session_with_serveable_world(1).await;
    let before = job_count(&session).await;

    let mut invalid = predictor_config();
    // `ContextPredictorTrainConfig::validate` refuses `context_k == 0`
    // (there must be at least one context point).
    invalid.context_k = 0;

    let error = session
        .train_context_predictor("episodes", &invalid)
        .await
        .expect_err("an invalid predictor config must be refused, never enqueued");
    assert!(
        matches!(error, JammiError::FineTune(_)),
        "the refusal must be typed, got {error:?}"
    );
    assert!(
        error.to_string().contains("context_k"),
        "the refusal must name the invalid field, got {error}"
    );
    assert_eq!(
        job_count(&session).await,
        before,
        "an invalid predictor config must enqueue nothing"
    );
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

/// One `Catalog::submit_job`/`submit_job_deduped` call site a REAL parse
/// (`syn::parse_file`, never grep/text) found in `crates/jammi-ai/src`'s
/// PRODUCTION code — a `#[cfg(test)]` module (the `mod tests { .. }`
/// blocks in `fine_tune/worker.rs`/`fine_tune/trainer.rs` that build a
/// placeholder job row directly, to drive the CLAIM/EXECUTE machinery under
/// test, never a submit edge a caller reaches) is never recursed into, so
/// no test fixture appears here — see
/// [`submit_call_sites_in_production_code`]'s own doc for why that scope
/// line is drawn there and not, say, at "every call in the crate".
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SubmitCallSite {
    file: String,
    enclosing_fn: String,
}

/// `#[cfg(test)]` exactly (a single bare `test` inside the `cfg(..)`
/// parens) — not `#[cfg(not(test))]`/`#[cfg(feature = "x")]`, which parse
/// as something other than one bare ident and so return `false` here,
/// leaving whatever they gate IN the scanned production surface (the
/// fail-closed direction: a body this predicate does not recognise as
/// test-only stays scanned, never silently exempted).
fn attr_is_cfg_test(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("cfg")
        && attr
            .parse_args::<syn::Ident>()
            .map(|id| id == "test")
            .unwrap_or(false)
}

/// [`syn::visit::Visit`] over one file's AST: records a [`SubmitCallSite`]
/// for every REFERENCE to `submit_job`/`submit_job_deduped` — a call or a
/// value — in the shapes the grammar allows: a method call
/// `x.submit_job(..)` (`visit_expr_method_call`); a path expression naming
/// the fn in ANY position, matched by the path's LAST segment under any
/// prefix or qualified self — the callee of `Catalog::submit_job(&c, ..)`,
/// `crate::db::Catalog::submit_job(..)`, `<Catalog>::submit_job(..)`, and
/// equally a fn-item captured as a value and invoked later
/// (`let route = Catalog::submit_job; route(&c, ..)`), handed to a
/// combinator (`.map(Catalog::submit_job)`) or stored in a field
/// (`visit_expr_path`, which fires for a path wherever it appears, so a
/// call position is not a special case); and a reference inside any macro
/// invocation's argument stream, `tokio::try_join!(c.submit_job(..))`,
/// which `syn` never descends into as an expression — `visit_macro` walks
/// the tokens through every nested group and records each exact `Ident`
/// spelled as either name (never a substring, never a string literal).
/// A value reference is recorded as a site because the fn it names can be
/// invoked anywhere afterwards; reviewing the reference is the only place
/// the review can happen. Each hit is tagged with its nearest enclosing
/// named `fn` — free function or `impl`/trait method alike. Never descends
/// into an item (a `mod`, a `fn`) carrying
/// `#[cfg(test)]` at all — see [`attr_is_cfg_test`] — which is what keeps
/// `fine_tune/worker.rs`'s and `fine_tune/trainer.rs`'s `mod tests { .. }`
/// fixture rows out of [`submit_call_sites_in_production_code`]'s universe.
struct SubmitCallScanner {
    file: String,
    fn_stack: Vec<String>,
    hits: Vec<SubmitCallSite>,
}

impl SubmitCallScanner {
    fn current_fn(&self) -> String {
        self.fn_stack
            .last()
            .cloned()
            .unwrap_or_else(|| "<no enclosing fn>".to_string())
    }
}

impl<'ast> syn::visit::Visit<'ast> for SubmitCallScanner {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.attrs.iter().any(attr_is_cfg_test) {
            return;
        }
        syn::visit::visit_item_mod(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if node.attrs.iter().any(attr_is_cfg_test) {
            return;
        }
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if node.attrs.iter().any(attr_is_cfg_test) {
            return;
        }
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.record_if_submit(&node.method.to_string());
        syn::visit::visit_expr_method_call(self, node);
    }

    /// A path expression in ANY position — the callee of a call, a value
    /// bound to a local, an argument, a struct field: only the LAST segment
    /// is the function name; a `<T>` qualified self lives in `qself`,
    /// outside the segments, and every prefix before the name is one this
    /// scan is indifferent to. `syn` reaches a call's callee through this
    /// same visitor, so a call is not a separate direction.
    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        // An inherent method is only ever spelled with an owner —
        // `Type::name`, `crate::m::Type::name`, `<Type>::name` — so a bare
        // single-segment path is a local or a free fn of that name, never
        // the catalog's method; recording it would review homonyms.
        if node.qself.is_some() || node.path.segments.len() >= 2 {
            if let Some(last) = node.path.segments.last() {
                self.record_if_submit(&last.ident.to_string());
            }
        }
        syn::visit::visit_expr_path(self, node);
    }

    /// A macro invocation in expression, statement or item position —
    /// `syn` routes all three here with the arguments as opaque tokens.
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        for name in submit_idents_in_tokens(node.tokens.clone()) {
            self.record_if_submit(&name);
        }
        syn::visit::visit_macro(self, node);
    }
}

impl SubmitCallScanner {
    fn record_if_submit(&mut self, name: &str) {
        if name == "submit_job" || name == "submit_job_deduped" {
            self.hits.push(SubmitCallSite {
                file: self.file.clone(),
                enclosing_fn: self.current_fn(),
            });
        }
    }
}

/// Every `Ident` token in `ts` (through every nested delimited group)
/// spelled `submit_job` or `submit_job_deduped`, in source order. A string
/// literal naming either is a `Literal` token, never an `Ident`.
fn submit_idents_in_tokens(ts: proc_macro2::TokenStream) -> Vec<String> {
    let mut out = Vec::new();
    for tt in ts {
        match tt {
            proc_macro2::TokenTree::Ident(i) => {
                let s = i.to_string();
                if s == "submit_job" || s == "submit_job_deduped" {
                    out.push(s);
                }
            }
            proc_macro2::TokenTree::Group(g) => out.extend(submit_idents_in_tokens(g.stream())),
            _ => {}
        }
    }
    out
}

/// The scan over one file's SOURCE TEXT — what
/// [`submit_call_sites_in_production_code`] runs on every tracked file and
/// what the `submit_shape_*` falsifications run on a synthetic fixture, so
/// a fixture exercises exactly the visitor the real gate uses.
fn scan_submit_source(file: &str, text: &str) -> Vec<SubmitCallSite> {
    let parsed = syn::parse_file(text)
        .unwrap_or_else(|e| panic!("{file}: could not parse as Rust source: {e}"));
    let mut scanner = SubmitCallScanner {
        file: file.to_string(),
        fn_stack: Vec::new(),
        hits: Vec::new(),
    };
    syn::visit::Visit::visit_file(&mut scanner, &parsed);
    scanner.hits
}

/// Every [`SubmitCallSite`] in every `.rs` file cargo COMPILES outside a
/// test target — the one universe
/// [`jammi_test_utils::source_universe::compiled_non_test_rs_files`] defines
/// and the raw byte-delete oracle in `jammi-db` shares: every workspace
/// member's `src/` (the crates AND the `ci/tools/*` members), every
/// `build.rs`, every `examples/` and `benches/` target; not `tests/`
/// directories. `#[cfg(test)]`
/// items inside those files are skipped by the scanner. The enumerated
/// universe [`every_submit_job_call_in_production_code_is_the_seam_or_a_reviewed_non_training_site`]
/// checks against its allow-list. `Catalog::submit_job`/`submit_job_deduped`
/// and `SubmitJobParams` are `pub`, so a caller anywhere cargo compiles is
/// in scope (`jammi-server`, `jammi-ballista` and `jammi-bench` hold
/// `Catalog` handles today, and `jammi-bench` has an example target); a
/// universe narrower than that would let a hand-built training-kind
/// submit pass unseen. The file list comes from `git ls-files`, never a
/// hand-maintained walk.
fn submit_call_sites_in_production_code() -> Vec<SubmitCallSite> {
    let root = jammi_test_utils::source_universe::repo_root();
    let files = jammi_test_utils::source_universe::compiled_non_test_rs_files(&root);
    let mut hits = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {file}: {e}"));
        hits.extend(scan_submit_source(file, &text));
    }
    hits.sort();
    hits
}

/// Every `Catalog::submit_job`/`submit_job_deduped`
/// reference in any compiled non-test `.rs` file in the workspace is on this exact,
/// reviewed allow-list — the universe is derived from a REAL parse of every
/// tracked file ([`submit_call_sites_in_production_code`]), not from a
/// hand-picked list of known functions: a fourth edge added ANYWHERE in the
/// crate, in a file this test never named, still shows up as an entry the allow-list does not
/// contain.
///
/// The allow-list, and why each entry is sound:
///
/// - `fine_tune/spec.rs::submit_admitted_training` — the training seam:
///   its ONLY caller-visible parameter that can produce a
///   training-kind row is `admitted: &AdmittedTrainingSpec`, a type whose
///   single field is private to this module, so the ONLY way any caller —
///   in this crate, or across the `jammi-bench` crate boundary, since this
///   function and `admit_training_spec` are both `pub` — can ever call this
///   with a training spec is by having called `admit_training_spec` first.
///   That is structural, not textual: see this test's own executed
///   falsification below of "does this actually block a bypass", and
///   `AdmittedTrainingSpec`'s own doc.
/// - `jobs.rs::enqueue` — its one call sits inside the `None` arm of
///   `match spec.as_training_spec() { Some(training) => { .. calls the
///   seam .. } None => { .. this call .. } }`: structurally unreachable
///   for a training kind, since that arm only runs when the training
///   projection returned nothing.
/// - `jobs.rs::run_now` — takes `spec: ComputeSpec` (never `TrainingSpec`)
///   as its OWN parameter type: the signature itself proves this call can
///   never carry a training kind, independent of what the body does.
/// - `jammi-db/src/catalog/jobs_repo.rs::submit_job` — the catalog's own
///   `submit_job` forwarding to `submit_job_deduped(p, None)`: the
///   definition side of the seam, inside the crate that owns the `jobs`
///   table. It mints no spec of its own; whatever reaches it already came
///   through one of the three `jammi-ai` sites above, which are the only
///   production callers of the catalog in the workspace.
/// - `jammi-client/src/lib.rs::submit_fine_tune` — a `submit_job` by NAME
///   only: the generated gRPC client's `JobService::submit_job` RPC, called
///   with a `SubmitJobRequest`, which lands in `jammi-server`'s handler and
///   from there in `jammi-ai`'s `enqueue` (a reviewed site above). It never
///   touches a `Catalog`; the scan keys on the identifier, so the homonym is
///   reviewed here rather than special-cased out of the universe.
///
/// RED (executed, reverted, never shipped): adding a FOURTH call —
/// `catalog.submit_job(SubmitJobParams { kind: "fine_tune", .. })` built by
/// hand inside a brand-new `pub(crate) async fn submit_training_spec_unadmitted`
/// in `crate::jobs` (a file already on the allow-list, but under a NAME the
/// list does not contain) — makes this test fail, printing the new
/// `(file, fn)` pair `("crates/jammi-ai/src/jobs.rs",
/// "submit_training_spec_unadmitted")` as an entry the allow-list does not
/// authorize — a fourth, unadmitted training edge, caught by this oracle.
/// Run the real scanner over a synthetic fixture and return `fn` names —
/// what every `submit_shape_*` falsification below asserts on.
fn submit_shape_fns(src: &str) -> Vec<String> {
    scan_submit_source("fixture.rs", src)
        .into_iter()
        .map(|h| h.enclosing_fn)
        .collect()
}

#[test]
fn submit_shape_1_a_method_call_is_found() {
    let src = "async fn f(c: C, p: P) { c.submit_job(p).await.unwrap(); }";
    assert_eq!(submit_shape_fns(src), vec!["f"]);
}

#[test]
fn submit_shape_2_a_path_call_is_found_under_any_prefix_and_qualified_self() {
    let bare = "async fn f(c: C, p: P) { Catalog::submit_job(&c, p).await; }";
    let qualified =
        "async fn f(c: C, p: P) { crate::db::Catalog::submit_job_deduped(&c, p, None).await; }";
    let qself = "async fn f(c: C, p: P) { <Catalog>::submit_job(&c, p).await; }";
    for (name, src) in [("bare", bare), ("qualified", qualified), ("qself", qself)] {
        assert_eq!(
            submit_shape_fns(src),
            vec!["f"],
            "submit shape 2 ({name}): a path-call spelling must be found"
        );
    }
}

#[test]
fn submit_shape_3_a_call_inside_a_macro_invocation_is_found_per_occurrence() {
    let joined = "async fn f(c: C, a: P, b: P) { tokio::try_join!(c.submit_job(a), c.submit_job_deduped(b, None)).unwrap(); }";
    assert_eq!(submit_shape_fns(joined), vec!["f", "f"]);
    let nested = "async fn f(c: C, p: P) { assert!(matches!(c.submit_job(p).await, Ok(()))); }";
    assert_eq!(submit_shape_fns(nested), vec!["f"]);
}

#[test]
fn submit_shape_4_a_path_captured_as_a_value_is_found_wherever_it_appears() {
    let captured =
        "async fn f(c: C, p: P) { let route = Catalog::submit_job; route(&c, p).await; }";
    let combinator =
        "fn f(c: C, ps: Vec<P>) { let _ = ps.into_iter().map(Catalog::submit_job_deduped); }";
    let field = "fn f() -> Routes { Routes { submit: <Catalog>::submit_job } }";
    for (name, src) in [
        ("captured", captured),
        ("combinator", combinator),
        ("field", field),
    ] {
        assert_eq!(
            submit_shape_fns(src),
            vec!["f"],
            "submit shape 4 ({name}): a path naming the fn as a VALUE must be found"
        );
    }
}

#[test]
fn submit_shape_controls_a_near_miss_identifier_or_a_string_literal_is_not_a_call() {
    let src = "async fn f(c: C, p: P) { c.submit_jobs(p).await; let _ = submit_job_count(); \
               tracing::warn!(\"submit_job refused\"); format!(\"submit_job_deduped\"); }";
    assert_eq!(submit_shape_fns(src), Vec::<String>::new());
}

#[test]
fn every_submit_job_call_in_production_code_is_the_seam_or_a_reviewed_non_training_site() {
    let allow: &[(&str, &str)] = &[
        (
            "crates/jammi-ai/src/fine_tune/spec.rs",
            "submit_admitted_training",
        ),
        ("crates/jammi-ai/src/jobs.rs", "enqueue"),
        ("crates/jammi-ai/src/jobs.rs", "run_now"),
        ("crates/jammi-db/src/catalog/jobs_repo.rs", "submit_job"),
        ("crates/jammi-client/src/lib.rs", "submit_fine_tune"),
    ];

    let hits = submit_call_sites_in_production_code();

    let mut unexpected = Vec::new();
    let mut counts: std::collections::BTreeMap<(&str, &str), usize> =
        allow.iter().map(|&(f, n)| ((f, n), 0usize)).collect();
    for hit in &hits {
        let key = allow
            .iter()
            .find(|&&(f, n)| f == hit.file && n == hit.enclosing_fn);
        match key {
            Some(&(f, n)) => {
                *counts.get_mut(&(f, n)).unwrap() += 1;
            }
            None => unexpected.push(format!("{}::{}", hit.file, hit.enclosing_fn)),
        }
    }

    assert!(
        unexpected.is_empty(),
        "a `submit_job`/`submit_job_deduped` call exists outside the reviewed allow-list — a \
         new training submit edge, or a review of this list, is needed: {unexpected:?}"
    );
    for (&(file, name), &count) in &counts {
        assert_eq!(
            count, 1,
            "{file}::{name} is on the allow-list with exactly one reviewed call, found {count}"
        );
    }
    assert_eq!(
        hits.len(),
        allow.len(),
        "the enumerated universe must equal the allow-list exactly: {hits:?}"
    );
}

/// The context-predictor edge and `submit_fine_tune_spec_deduped` are
/// NOT on the allow-list above: both reach
/// `submit_admitted_training` themselves rather than calling
/// `Catalog::submit_job_deduped` directly, so their OWN admission is pinned
/// by this behavioural check — each calls
/// `crate::fine_tune::spec::admit_training_spec` in its own body before
/// calling the seam, source-verified over the two files whose training-kind row NEVER touches
/// `Catalog::submit_job`/`submit_job_deduped` directly at all.
///
/// Mutation (executed and reverted, never shipped): deleting the
/// `admit_training_spec` call from `train_context_predictor_deduped` drops
/// this file's own call count to 0 and this test fails, naming the file.
#[test]
fn the_two_seam_calling_edges_admit_before_calling_the_seam() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-ai has two ancestors: crates/, then the repo root")
        .to_path_buf();

    let edges: &[(&str, &str)] = &[
        (
            "crates/jammi-ai/src/session.rs",
            "async fn submit_fine_tune_spec_deduped(",
        ),
        (
            "crates/jammi-ai/src/pipeline/context_predictor.rs",
            "pub(crate) async fn train_context_predictor_deduped(",
        ),
    ];

    const ADMIT_CALL: &str = "fine_tune::spec::admit_training_spec(";
    const SEAM_CALL: &str = "fine_tune::spec::submit_admitted_training(";

    for (path, fn_sig) in edges {
        let full = root.join(path);
        let src =
            std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("could not read {path}: {e}"));
        let fn_start = src.find(fn_sig).unwrap_or_else(|| {
            panic!("{path} no longer defines `{fn_sig}` — this oracle's edge list is stale")
        });
        let after_sig = &src[fn_start..];
        let body_end = after_sig.find("\n    }\n").unwrap_or(after_sig.len());
        let body = &after_sig[..body_end];
        assert_eq!(
            body.matches(ADMIT_CALL).count(),
            1,
            "{path}'s `{fn_sig}` must call `{ADMIT_CALL}` exactly once before writing a jobs row"
        );
        assert_eq!(
            body.matches(SEAM_CALL).count(),
            1,
            "{path}'s `{fn_sig}` must call `{SEAM_CALL}` exactly once — the seam, not a hand-built \
             `SubmitJobParams`"
        );
    }
}
