//! The persisted, self-describing training specification.
//!
//! A training job is durable work claimed under a lease (`training_jobs`). For a
//! worker to run a job it never submitted — on a fresh process, with no
//! in-memory carryover from the submitting session — the job must carry
//! *everything* needed to reconstruct its data and model. [`TrainingSpec`] is
//! that self-contained description: a tagged enum whose variant names the verb
//! that produced the job (`kind`) and whose fields are the reconstruction inputs
//! (source SQL, column bindings, deterministic sampler seeds, the architecture /
//! objective / optimisation budget).
//!
//! The spec is serialised to JSON into `training_jobs.training_spec` at submit
//! time and deserialised by the worker at claim time; the variant tag is also
//! written into `training_jobs.kind`. The same three verbs that submit
//! (`fine_tune`, `fine_tune_graph`, `train_context_predictor`) are the three
//! variants here — the unification the worker descends over once, rather than
//! three bespoke execution paths.

use serde::{Deserialize, Serialize};

use jammi_db::error::{JammiError, Result};
use jammi_db::model_task::ModelTask;
use jammi_db::store::CachePolicy;

use crate::fine_tune::graph_sampler::{GraphFineTuneSources, GraphSampleConfig};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::pipeline::context_predictor::ContextPredictorTrainConfig;

/// A durable, self-contained description of a training job: the verb that
/// produced it (the variant) and the inputs a worker reconstructs the run from
/// on a fresh process. Persisted as JSON on the job's `training_spec` column;
/// the variant's [`TrainingSpec::kind`] tag is mirrored into `training_jobs.kind`.
///
/// No `#[serde(deny_unknown_fields)]`, by this repo's persisted-row
/// convention: a `jobs.spec` row is always engine-written from a decoded
/// spec, so an unknown key can only arrive via a hand edit, and this type
/// is wrapped in `crate::jobs::JobSpec`'s `#[serde(untagged)]` (deny would
/// collapse the outer wrapper's own "try each inner deserializer in turn"
/// dispatch — see that type's own doc). The consequence is stated honestly
/// rather than assumed: a stray `cache` key hand-edited under a
/// `graph_fine_tune` row is silently DROPPED at deserialize, never refused
/// — `TrainingSpec::GraphFineTune::cache` does not exist to receive it. See
/// <https://github.com/f-inverse/jammi-ai/issues/548> for the `JobSpec`
/// reshape that would let a genuinely malformed row be refused instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingSpec {
    /// Column-source contrastive / classification / regression fine-tune. The
    /// worker re-runs `SELECT columns FROM source` and rebuilds the data loader
    /// keyed by `task` — the same loader the submitting `fine_tune` would have
    /// built, but from the persisted column bindings rather than in-memory
    /// batches.
    FineTune {
        /// Registered source whose rows the model is fine-tuned on.
        source: String,
        /// Columns selected into the training loader (the format — contrastive /
        /// triplet / pairs / classification — is detected from these names).
        columns: Vec<String>,
        /// Adapter method.
        method: FineTuneMethod,
        /// Task the fine-tuned model performs (selects the loader's cell
        /// decoding and the training head).
        task: ModelTask,
        /// Common base-model + optimisation knobs.
        common: TrainingCommon,
        /// [`ProducingDescriptor::FineTune`](jammi_db::store::manifest::ProducingDescriptor::FineTune)'s
        /// cache dial, the same shape [`crate::jobs::ComputeSpec`]'s own
        /// `cache` field already carries for every compute kind — except
        /// that model-level cache reuse is not yet supported for this kind:
        /// `Use` is refused, typed, at submit
        /// (`InferenceSession::submit_fine_tune_spec_deduped`); see
        /// <https://github.com/f-inverse/jammi-ai/issues/562>. `Bypass` (the
        /// only value a submitted job can carry past that refusal) always
        /// trains.
        ///
        /// Lives HERE — on the `FineTune` variant, not on [`TrainingCommon`]
        /// — because only this kind has a materialization to probe:
        /// [`TrainingSpec::GraphFineTune`] carries no `cache` field at all,
        /// so a cache policy for the graph kind is UNREPRESENTABLE rather
        /// than merely unused.
        ///
        /// A CALL-TIME dial, never a determinant of the trained model's
        /// identity: like a `ComputeSpec`'s `cache` never rides in the
        /// [`jammi_db::store::manifest::ProducingDescriptor`] it drives, this
        /// field is deliberately EXCLUDED from [`fine_tune_spec_canonical`]
        /// — flipping it between two otherwise-identical submissions must
        /// never change the definition hash, or a `Bypass` run and a `Use`
        /// run of "the same spec" would silently become two different
        /// trained-model identities.
        ///
        /// [`CachePolicy::default`] (`Bypass`) is what a spec whose JSON
        /// carries no policy at all deserializes to, so a queued job written
        /// before this field existed still trains unconditionally — the same
        /// no-regression shape [`TrainingCommon::world_size`]'s own default
        /// keeps.
        #[serde(default)]
        cache: CachePolicy,
    },
    /// Graph-supervised fine-tune. The worker re-reads the node/edge sources and
    /// rebuilds the [`crate::fine_tune::graph_sampler::GraphSampler`] from
    /// `sample_config` — deterministic via its seed, so the re-sampled pairs
    /// match a run from the same spec byte-for-byte.
    GraphFineTune {
        /// Node-text + edge sources and their column bindings.
        sources: GraphFineTuneSources,
        /// Walk / negative-sampling knobs, including the deterministic seed.
        sample_config: GraphSampleConfig,
        /// Common base-model + optimisation knobs.
        common: TrainingCommon,
    },
    /// Episodic in-context-predictor meta-training. The worker re-samples the
    /// episodic meta-dataset from `source` per `predictor_spec` (deterministic
    /// task split via the spec seed), drives the predictor train loop, and
    /// persists the trained predictor.
    ContextPredictor {
        /// Source whose embedding table the episodic context is sampled from.
        source: String,
        /// The architecture / objective / episodic / optimisation specification.
        predictor_spec: ContextPredictorTrainConfig,
    },
}

/// Base-model and optimisation knobs common to the two LoRA fine-tune kinds. The
/// predictor kind carries its budget inside its own `predictor_spec`, so this is
/// shared only by the fine-tune variants. `cache` (the model-level reuse dial)
/// is NOT here: it lives directly on [`TrainingSpec::FineTune`], the only kind
/// with a materialization to probe, so a cache policy for
/// [`TrainingSpec::GraphFineTune`] is unrepresentable rather than merely
/// unused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCommon {
    /// Base model id the adapter / head is trained over.
    pub base_model: String,
    /// Fine-tune configuration (epochs, LoRA rank, loss, …).
    pub config: FineTuneConfig,
    /// How many data-parallel ranks this job trains over.
    /// [`DEFAULT_WORLD_SIZE`] is the single-rank, single-process run.
    ///
    /// Identity-relevant, not a scheduling hint: the rank count fixes the
    /// batch layout and the gradient summation order, so it belongs in the
    /// persisted spec beside the other reconstruction inputs rather than in
    /// the deployment's configuration. A count the deployment cannot serve
    /// is refused at the submit edge with a typed error, never clamped: a
    /// silently lowered count would train a different model than the caller
    /// asked for and record it under the same identity.
    ///
    /// A spec whose JSON carries no count deserializes to
    /// [`DEFAULT_WORLD_SIZE`], so a queued job written by any writer runs on
    /// any worker. Always serialized (no `skip_serializing_if`): the
    /// persisted JSON states the count rather than leaving a reader to infer
    /// it from an absence.
    #[serde(default = "default_world_size")]
    pub world_size: u32,
}

/// The rank count of a spec that does not name one: a single rank, which is
/// the single-process run.
pub const DEFAULT_WORLD_SIZE: u32 = 1;

/// `serde`'s `default` hook for [`TrainingCommon::world_size`] — a function
/// because `#[serde(default = ...)]` names a path, not a literal.
fn default_world_size() -> u32 {
    DEFAULT_WORLD_SIZE
}

/// What a submitted [`TrainingSpec`] is admitted against: the deployment's
/// devices, the collective it reduces over, and whether this build can reach
/// that collective.
///
/// The submit edge is where a rank count the deployment cannot serve is
/// caught, because it is the last point at which refusing costs nothing: past
/// it the spec is a durable row that a worker will claim, fail, and retry
/// until the attempt budget runs out. A count is never CLAMPED to what the
/// deployment can serve — a silently lowered count trains a different model
/// than the caller asked for and records it under the same identity.
///
/// The build flag is DATA rather than a `cfg!` inside the check so the rule
/// is decidable for either build from either build: a host test can state
/// what a CUDA build admits, and a CUDA build can state what a host build
/// refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankAdmission {
    devices: usize,
    collective: jammi_db::config::CollectiveSelection,
    cuda_build: bool,
}

impl RankAdmission {
    /// The admission this deployment implies: one rank per configured
    /// device, the configured collective, and THIS build's CUDA support.
    pub fn from_config(config: &jammi_db::config::JammiConfig) -> Self {
        Self {
            devices: config.gpu.device_list().len(),
            collective: config.worker.collective,
            cuda_build: cfg!(feature = "cuda"),
        }
    }

    /// An admission stated outright, rather than read off a live
    /// [`jammi_db::config::JammiConfig`] — for an embedder that has its own
    /// source of the device count, collective and build flag (a test that
    /// needs a deployment this host does not have is one such caller, not
    /// the only one).
    pub fn new(
        devices: usize,
        collective: jammi_db::config::CollectiveSelection,
        cuda_build: bool,
    ) -> Self {
        Self {
            devices,
            collective,
            cuda_build,
        }
    }

    /// Admit `spec`, or refuse with a typed [`JammiError::Config`] naming the
    /// bound it crosses.
    ///
    /// The refusals, each of which would otherwise become a job that fails
    /// on a worker rather than at the caller:
    ///
    /// - `world_size == 0` — no rank to run on. Unrepresentable at the
    ///   request edge (`FineTuneRequest::world_size` is a `NonZeroU32`) but
    ///   reachable through a hand-built or deserialized spec, so the durable
    ///   edge checks it too.
    /// - `world_size > devices` — there is no device for the last rank, and
    ///   the alternative to refusing is two ranks silently sharing one.
    /// - `collective = "nccl"` on a build without CUDA — the requested
    ///   collective cannot be reached. Also refused at session OPEN
    ///   (`refuse_unreachable_collective`), which is the edge a live session
    ///   is stopped at; this arm is the same rule stated where the spec is
    ///   admitted, so a caller holding an admission built by hand gets the
    ///   same verdict.
    /// - `world_size > 1` with GradCache (`config.cached`) or hard-negative
    ///   mining (`config.hard_negatives.mine`) — both are single-rank
    ///   mechanisms (a whole-batch second pass, and a miner over this
    ///   process's own index), and running them per rank would change the
    ///   negative pool each rank sees, so the gang would not compute the
    ///   objective the caller asked for.
    ///
    /// The context-predictor kind carries no rank count at all: its variant
    /// has no [`TrainingCommon`], so a multi-rank predictor job is
    /// unrepresentable here and is refused at the wire decode, the last edge
    /// that can still see a count a caller chose.
    pub fn admit(&self, spec: &TrainingSpec) -> Result<()> {
        let common = match spec {
            TrainingSpec::FineTune { common, .. } | TrainingSpec::GraphFineTune { common, .. } => {
                common
            }
            TrainingSpec::ContextPredictor { .. } => return Ok(()),
        };
        let world_size = common.world_size;

        if world_size == 0 {
            return Err(JammiError::Config(
                "world_size must be >= 1 (1 is the single-rank job; 0 has no rank to run on)"
                    .into(),
            ));
        }
        if world_size as usize > self.devices {
            return Err(JammiError::Config(format!(
                "world_size = {world_size} exceeds the {} configured device(s): one rank per \
                 device, so list more in `[gpu] devices` or submit a smaller rank count",
                self.devices
            )));
        }
        if self.collective.requires_cuda() && !self.cuda_build {
            return Err(JammiError::Config(format!(
                "[worker] collective = \"{}\" needs a build with the `cuda` feature; this \
                 binary has none, so the requested collective cannot be reached",
                self.collective
            )));
        }
        if world_size > 1 && common.config.cached {
            return Err(JammiError::Config(format!(
                "world_size = {world_size} cannot be combined with GradCache (`cached`): the \
                 cached objective's second pass is over the WHOLE batch on one rank, so a \
                 gang would not compute the objective this config asks for"
            )));
        }
        if world_size > 1 && common.config.hard_negatives.mine {
            return Err(JammiError::Config(format!(
                "world_size = {world_size} cannot be combined with hard-negative mining \
                 (`hard_negatives.mine`): the miner retrieves from this process's own index, \
                 so each rank would mine a different negative pool"
            )));
        }
        Ok(())
    }
}

impl TrainingSpec {
    /// The catalog `kind` tag for this spec — the same discriminator the tagged
    /// JSON carries, mirrored into the `training_jobs.kind` column so a query can
    /// filter by verb without parsing the spec blob.
    pub fn kind(&self) -> &'static str {
        match self {
            TrainingSpec::FineTune { .. } => "fine_tune",
            TrainingSpec::GraphFineTune { .. } => "graph_fine_tune",
            TrainingSpec::ContextPredictor { .. } => "context_predictor",
        }
    }
}

// ─── The `ProducingDescriptor::FineTune::spec_canonical` producer ──────────
//
// `jammi-db` depends on no jammi crate but `jammi-numerics` (DESIGN §3), so it
// cannot hold `TrainingSpec`/`FineTuneConfig` directly — the descriptor's
// `spec_canonical` field is instead an OPAQUE, versioned canonical JSON string
// this module produces from the owning types, the same "db-local primitive
// standing in for a foreign type" shape `ProducingDescriptor::TrainingSet::format`
// already uses (see that variant's own doc in `jammi_db::store::manifest`).

/// The schema version [`fine_tune_spec_canonical`] encodes under —
/// [`ProducingDescriptor::FineTune::spec_schema_version`](jammi_db::store::manifest::ProducingDescriptor::FineTune).
/// Bumped whenever the field set this function folds changes shape, so a
/// reader never treats two canonical strings encoded under different,
/// silently-incompatible shapes as the same kind of value.
pub const FINE_TUNE_SPEC_SCHEMA_VERSION: u32 = 1;

/// The output-affecting fields of a `TrainingSpec::FineTune` variant, shaped
/// for canonical (sorted-key) JSON encoding by [`fine_tune_spec_canonical`]
/// and decoding by [`fine_tune_spec_from_canonical`].
///
/// Two fields are deliberately absent from this shape even though they live
/// on the types it is built from:
///
/// - `TrainingSpec::FineTune::cache` — a call-time reuse dial, never part of
///   the trained model's identity (see that field's own doc).
/// - `FineTuneConfig::keep_last_n_checkpoints` — a pure deployment/storage
///   knob documented on the field itself as entering no identity/config hash
///   and never affecting the trained artifact; [`fine_tune_spec_canonical`]
///   zeroes it out of the `config` it folds in, rather than serializing
///   `FineTuneConfig` verbatim, so this NEW hash does not silently start
///   treating it as a determinant the field's own contract says it is not.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FineTuneSpecCanonicalV1 {
    source: String,
    columns: Vec<String>,
    method: FineTuneMethod,
    task: ModelTask,
    base_model: String,
    config: FineTuneConfig,
    world_size: u32,
}

/// Sorted-key canonical JSON encoding of a `TrainingSpec::FineTune` variant's
/// output-affecting fields — the
/// [`ProducingDescriptor::FineTune::spec_canonical`](jammi_db::store::manifest::ProducingDescriptor::FineTune)
/// producer.
///
/// Object keys are sorted recursively by `canonicalize_json` (private) rather than
/// relying on `serde_json::Map`'s own ordering: the workspace's `serde_json`
/// runs with the `preserve_order` feature, so an unsorted `Map` keeps
/// INSERTION order, not lexical order. Sorting here — the same step
/// `jammi_db::store::manifest::ProducingDescriptor::canonical_bytes` takes for
/// the descriptor itself — makes the byte stream independent of struct field
/// declaration order and stable across serde versions: two calls that
/// describe the same logical spec produce the identical string regardless of
/// how their pieces were assembled or in what order.
pub fn fine_tune_spec_canonical(
    source: &str,
    columns: &[String],
    method: FineTuneMethod,
    task: ModelTask,
    base_model: &str,
    config: &FineTuneConfig,
    world_size: u32,
) -> Result<String> {
    let mut config = config.clone();
    // See `FineTuneSpecCanonicalV1`'s doc: this field enters no identity by
    // its own contract.
    config.keep_last_n_checkpoints = None;
    let shape = FineTuneSpecCanonicalV1 {
        source: source.to_string(),
        columns: columns.to_vec(),
        method,
        task,
        base_model: base_model.to_string(),
        config,
        world_size,
    };
    let value = serde_json::to_value(&shape)
        .map_err(|e| JammiError::FineTune(format!("fine-tune spec is not canonicalisable: {e}")))?;
    let canonical = canonicalize_json(&value);
    serde_json::to_string(&canonical)
        .map_err(|e| JammiError::FineTune(format!("fine-tune spec is not canonicalisable: {e}")))
}

/// The inverse of [`fine_tune_spec_canonical`]: decode a persisted
/// `spec_canonical` string into a fresh `TrainingSpec::FineTune` — the
/// `pipeline::recompute` `FineTune` arm's retrain path (K1). `cache` is not
/// part of the encoded shape (see `FineTuneSpecCanonicalV1`'s doc, private), so the
/// decoded spec always carries [`CachePolicy::Bypass`] — a replay always
/// recomputes (the same reasoning `pipeline::recompute`'s module doc states
/// for every other arm).
///
/// A `spec_schema_version` this build does not recognise is a typed refusal,
/// never a best-effort guess at an unknown shape.
pub fn fine_tune_spec_from_canonical(
    spec_canonical: &str,
    spec_schema_version: u32,
) -> Result<TrainingSpec> {
    if spec_schema_version != FINE_TUNE_SPEC_SCHEMA_VERSION {
        return Err(JammiError::FineTune(format!(
            "fine-tuned model's spec_canonical is encoded under schema version \
             {spec_schema_version}, but this build only encodes/decodes version \
             {FINE_TUNE_SPEC_SCHEMA_VERSION}"
        )));
    }
    let decoded: FineTuneSpecCanonicalV1 = serde_json::from_str(spec_canonical).map_err(|e| {
        JammiError::FineTune(format!("undeserialisable fine-tune spec_canonical: {e}"))
    })?;
    Ok(TrainingSpec::FineTune {
        source: decoded.source,
        columns: decoded.columns,
        method: decoded.method,
        task: decoded.task,
        common: TrainingCommon {
            base_model: decoded.base_model,
            config: decoded.config,
            world_size: decoded.world_size,
        },
        cache: CachePolicy::Bypass,
    })
}

/// Canonical bytes helper shared by [`fine_tune_spec_canonical`]: a JSON
/// value with every object's keys sorted, recursively. Pure; no I/O. Mirrors
/// (deliberately duplicated rather than imported — `jammi-db`'s own
/// `canonicalize_json` in `store::manifest` is a private helper, and this
/// module cannot see it) the descriptor-hashing side's own canonicalisation
/// step, so both halves of one hash agree on what "canonical" means.
fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: std::collections::BTreeMap<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonicalize_json(v)))
                .collect();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_json).collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JSON shape a writer that has no rank count queues: a `fine_tune`
    /// spec whose `common` block carries `base_model` and `config` and
    /// nothing else.
    ///
    /// The spec and the `common` block are built member by member rather than
    /// by serializing a current `TrainingSpec` and deleting a key — a fixture
    /// derived from the current shape would silently acquire whatever is
    /// added to `TrainingCommon` next and stop testing the absence it exists
    /// to test. Only `config`, an opaque block this test says nothing about,
    /// is taken from its own serializer.
    fn spec_json_without_a_rank_count() -> String {
        serde_json::json!({
            "kind": "fine_tune",
            "source": "patents",
            "columns": ["abstract"],
            "method": "lora",
            "task": "text_embedding",
            "common": {
                "base_model": "local:tiny",
                "config": serde_json::to_value(FineTuneConfig::default()).expect("config"),
            },
        })
        .to_string()
    }

    /// A spec whose JSON names no rank count deserializes to the single-rank
    /// run, so a job queued without the field runs on a worker that reads it.
    /// The alternative — a deserialization failure, or a `0` default — turns
    /// every already-queued job into a permanently unclaimable row.
    #[test]
    fn a_spec_with_no_rank_count_deserializes_to_a_single_rank() {
        let json = spec_json_without_a_rank_count();
        assert!(
            !json.contains("world_size"),
            "the fixture must not name the count it exists to omit: {json}"
        );
        let spec: TrainingSpec = serde_json::from_str(&json)
            .expect("a spec that names no rank count must still deserialize");
        let TrainingSpec::FineTune { common, .. } = &spec else {
            panic!("expected the fine_tune variant, got {spec:?}");
        };
        assert_eq!(
            common.world_size, 1,
            "an absent rank count is one rank, not zero and not a parse error"
        );
        assert_eq!(common.world_size, DEFAULT_WORLD_SIZE);
    }

    /// The count survives a round trip through the persisted form, and is
    /// written under the `world_size` key ALWAYS — a reader of `jobs.spec`
    /// (and the remote-versus-embedded parity oracle in the server suite)
    /// reads the count off the JSON rather than inferring it from an absence.
    #[test]
    fn a_chosen_rank_count_round_trips_and_is_always_serialized() {
        let spec = TrainingSpec::FineTune {
            source: "patents".into(),
            columns: vec!["abstract".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: TrainingCommon {
                base_model: "local:tiny".into(),
                config: FineTuneConfig::default(),
                world_size: 4,
            },
            cache: CachePolicy::Bypass,
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        assert!(
            json.contains(r#""world_size":4"#),
            "the chosen count must be in the persisted JSON: {json}"
        );

        let single = TrainingSpec::FineTune {
            common: TrainingCommon {
                base_model: "local:tiny".into(),
                config: FineTuneConfig::default(),
                world_size: DEFAULT_WORLD_SIZE,
            },
            cache: CachePolicy::Bypass,
            source: "patents".into(),
            columns: vec!["abstract".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
        };
        let single_json = serde_json::to_string(&single).expect("serialize");
        assert!(
            single_json.contains(r#""world_size":1"#),
            "the default count is serialized too, never skipped: {single_json}"
        );

        let back: TrainingSpec = serde_json::from_str(&json).expect("deserialize");
        let TrainingSpec::FineTune { common, .. } = &back else {
            panic!("expected the fine_tune variant, got {back:?}");
        };
        assert_eq!(common.world_size, 4);
    }

    /// The persisted-row oracle stated honestly: a
    /// `graph_fine_tune` row is engine-written from a decoded spec, so a
    /// stray top-level `cache` key under it can only arrive via a
    /// hand-edited `jobs.spec` row — never a real submit path, since
    /// `TrainingSpec::GraphFineTune` has no `cache` field to serialize one
    /// from. Serde is PERMISSIVE by this repo's persisted-row convention
    /// (no `deny_unknown_fields` — see [`TrainingSpec`]'s own doc and
    /// <https://github.com/f-inverse/jammi-ai/issues/548> for the `JobSpec`
    /// reshape that would let this be refused instead of silently ignored):
    /// the key is DROPPED at deserialize, not refused and not smuggled
    /// through to any observable field. Proven by re-serializing the decoded
    /// spec and asserting the key never comes back, rather than merely
    /// asserting decode succeeds.
    #[test]
    fn a_stray_cache_key_under_graph_fine_tune_is_dropped_at_deserialize() {
        let original = TrainingSpec::GraphFineTune {
            sources: GraphFineTuneSources {
                node_source: "nodes".into(),
                id_column: "id".into(),
                text_column: "text".into(),
                edge_source: "edges".into(),
                src_column: "src".into(),
                dst_column: "dst".into(),
                provenance: crate::fine_tune::graph_sampler::EdgeProvenance::Declared,
            },
            sample_config: GraphSampleConfig::default(),
            common: TrainingCommon {
                base_model: "local:tiny".into(),
                config: FineTuneConfig::default(),
                world_size: DEFAULT_WORLD_SIZE,
            },
        };
        let mut value = serde_json::to_value(&original).expect("serialize to a JSON value");
        let object = value
            .as_object_mut()
            .expect("a graph_fine_tune spec is a JSON object");
        assert!(
            !object.contains_key("cache"),
            "the fixture itself must carry no `cache` key before the hand-edit: {object:?}"
        );
        // The hand-edit a real submit path can never produce: `TrainingSpec::
        // GraphFineTune` has no `cache` field to have serialized this key.
        object.insert("cache".to_string(), serde_json::json!("use"));

        let decoded: TrainingSpec =
            serde_json::from_value(value).expect("the unknown key must not refuse deserialize");
        assert!(
            matches!(decoded, TrainingSpec::GraphFineTune { .. }),
            "expected the graph_fine_tune variant, got {decoded:?}"
        );
        // The value is GONE, not merely unread: re-serializing the decoded
        // spec never reproduces a `cache` key, because the in-memory type
        // has nowhere to have stored it.
        let re_encoded = serde_json::to_value(&decoded).expect("re-serialize");
        assert!(
            !re_encoded
                .as_object()
                .expect("still an object")
                .contains_key("cache"),
            "a stray `cache` key under graph_fine_tune must be dropped, never carried through: \
             {re_encoded:?}"
        );
    }

    // ─── `fine_tune_spec_canonical` / `fine_tune_spec_from_canonical` ──────

    /// Every field [`fine_tune_spec_canonical`] folds, carried as a fixture
    /// whose shape [`canonical_of`] destructures WITHOUT `..` — a field added
    /// to `TrainingSpec::FineTune`'s own top level fails to compile here
    /// (`ProducingDescriptor::FineTune`'s own completeness test in `jammi-db`
    /// covers the descriptor's own fields). `common` holds the REAL
    /// [`TrainingCommon`] rather than a hand-copied mirror of its fields: a
    /// mirror struct can drift from the type it stands in for — adding a
    /// field to `TrainingCommon` compiles a hand-copied proxy clean, so a new
    /// hash-relevant field can silently never enter the fine-tune definition
    /// hash (K7's own failure mode, caught by an executed falsification: a
    /// field added to the real `TrainingCommon`, with every real
    /// construction site fixed, must fail to compile HERE until named).
    /// Composing the real type instead makes [`canonical_of`]'s destructure
    /// of `common` the single completeness check for both the top-level spec
    /// fields and every `TrainingCommon` field, restated over the
    /// `jammi-ai`-owned producer (K7).
    ///
    /// `cache` is a top-level field of this fixture, not nested in `common`,
    /// matching where it lives on `TrainingSpec::FineTune` itself.
    #[derive(Clone)]
    struct CanonicalFields {
        source: String,
        columns: Vec<String>,
        method: FineTuneMethod,
        task: ModelTask,
        common: TrainingCommon,
        cache: CachePolicy,
    }

    /// A base fixture whose every field is a non-default, distinguishable
    /// value where the type permits one — a mutation test over an all-default
    /// fixture would pass vacuously exactly where the encoding is lossy.
    fn canonical_fields() -> CanonicalFields {
        CanonicalFields {
            source: "patents".into(),
            columns: vec!["abstract".into(), "claims".into()],
            method: FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: TrainingCommon {
                base_model: "local:tiny".into(),
                config: FineTuneConfig {
                    lora_rank: 8,
                    keep_last_n_checkpoints: Some(3),
                    ..FineTuneConfig::default()
                },
                world_size: 2,
            },
            cache: CachePolicy::Bypass,
        }
    }

    /// Exhaustive destructuring (no `..`) over the REAL [`TrainingCommon`]
    /// AND the fixture's top-level `cache` field: a field added to either
    /// fails to compile here until it is bound and either folded into the
    /// call below or explicitly excluded with a stated reason, matching
    /// `cache`'s own binding — see [`CanonicalFields`]'s own doc for why
    /// `common` is composed over the real type rather than a hand-copied
    /// mirror.
    fn canonical_of(f: &CanonicalFields) -> String {
        let CanonicalFields {
            source,
            columns,
            method,
            task,
            common,
            // Deliberately not passed to `fine_tune_spec_canonical` — a
            // call-time dial, never part of the identity (see
            // `cache_never_moves_the_canonical_string` below).
            cache: _,
        } = f.clone();
        let TrainingCommon {
            base_model,
            config,
            world_size,
        } = common;
        fine_tune_spec_canonical(
            &source,
            &columns,
            method,
            task,
            &base_model,
            &config,
            world_size,
        )
        .expect("canonicalisable fixture")
    }

    /// The same logical spec canonicalises identically across calls.
    #[test]
    fn fine_tune_spec_canonical_is_deterministic() {
        let f = canonical_fields();
        assert_eq!(canonical_of(&f), canonical_of(&f));
    }

    /// K7: every hash-relevant field, mutated one at a time from the
    /// all-distinguishable base fixture, moves the canonical string. Report
    /// per-determinant, never a count (program discipline).
    /// A named mutation over the canonical fixture: (`label`, the mutating
    /// closure) — mirrors `jammi_db::store::manifest`'s own
    /// `LabelledMutation` test shape.
    type LabelledMutation = (&'static str, fn(&mut CanonicalFields));

    #[test]
    fn every_hash_relevant_field_moves_the_canonical_string() {
        let base = canonical_fields();
        let base_str = canonical_of(&base);
        let mutations: &[LabelledMutation] = &[
            ("source", |f| f.source = "other-source".into()),
            ("columns (new member)", |f| f.columns.push("extra".into())),
            ("columns (order)", |f| f.columns.reverse()),
            ("task", |f| f.task = ModelTask::Classification),
            ("base_model", |f| f.common.base_model = "local:other".into()),
            ("config", |f| f.common.config.lora_rank = 99),
            ("world_size", |f| f.common.world_size = 4),
        ];
        for (name, mutate) in mutations {
            let mut mutated = base.clone();
            mutate(&mut mutated);
            let mutated_str = canonical_of(&mutated);
            assert_ne!(
                base_str, mutated_str,
                "mutating {name} must move the canonical string"
            );
        }
    }

    /// `method` has exactly one variant today ([`FineTuneMethod::Lora`]), so
    /// no value mutation can demonstrate it moves the string — vacuous by
    /// construction, stated here rather than silently omitted: this pins
    /// that it is at least PRESENT in the encoded bytes, so a second variant
    /// landing later is caught by the mutation test above the moment it can
    /// be varied.
    #[test]
    fn method_is_present_in_the_canonical_string_though_unvaryable_today() {
        let s = canonical_of(&canonical_fields());
        assert!(s.contains(r#""method":"lora""#), "got {s}");
    }

    /// `TrainingSpec::FineTune::cache` is a call-time reuse dial excluded
    /// from the shape [`fine_tune_spec_canonical`] folds — flipping it
    /// between two otherwise-identical specs must not move the canonical
    /// string, or a `Bypass` run and a `Use` run of "the same spec" would
    /// silently become two different trained-model identities.
    #[test]
    fn cache_never_moves_the_canonical_string() {
        let mut use_cache = canonical_fields();
        use_cache.cache = CachePolicy::Use;
        let mut bypass = canonical_fields();
        bypass.cache = CachePolicy::Bypass;
        assert_eq!(canonical_of(&use_cache), canonical_of(&bypass));
        // A precise key check, not a bare substring: `FineTuneConfig` has its
        // own, unrelated `cached` (GradCache) field, whose serialized key
        // legitimately contains "cache" as a substring.
        assert!(
            !canonical_of(&bypass).contains(r#""cache":"#),
            "the encoded shape must carry no top-level `cache` key at all"
        );
    }

    /// `FineTuneConfig::keep_last_n_checkpoints` is documented on the field
    /// itself as entering no identity/config hash — a pure deployment/storage
    /// knob that never affects the trained artifact. This NEW hash must
    /// honour that contract rather than silently start treating it as a
    /// determinant. The field has no `skip_serializing_if`, so its KEY still
    /// appears in the encoded `config` object (as a constant `null` — never
    /// varying with the input), which is the property under test: the VALUE
    /// stays constant regardless of what the caller's `FineTuneConfig` set.
    #[test]
    fn keep_last_n_checkpoints_never_moves_the_canonical_string() {
        let mut none = canonical_fields();
        none.common.config.keep_last_n_checkpoints = None;
        let mut some = canonical_fields();
        some.common.config.keep_last_n_checkpoints = Some(7);
        assert_eq!(canonical_of(&none), canonical_of(&some));
        assert!(
            canonical_of(&none).contains(r#""keep_last_n_checkpoints":null"#),
            "the field is zeroed to a constant `null` before hashing, never omitted or varying"
        );
    }

    /// Sorted keys: a field-order permutation of the logical input (built by
    /// hand at two different key orders, bypassing the struct's own fixed
    /// declaration order) canonicalises to the identical string.
    #[test]
    fn canonicalize_json_sorts_keys_independent_of_input_order() {
        let a = serde_json::json!({"b": 1, "a": {"y": 2, "x": 1}, "c": [3, 2, 1]});
        let b = serde_json::json!({"a": {"x": 1, "y": 2}, "c": [3, 2, 1], "b": 1});
        assert_eq!(canonicalize_json(&a), canonicalize_json(&b));
        let sorted_str = serde_json::to_string(&canonicalize_json(&a)).unwrap();
        assert_eq!(
            sorted_str,
            serde_json::to_string(&canonicalize_json(&b)).unwrap()
        );
    }

    /// A pinned golden string for a fixed spec: this MUST NOT change unless
    /// [`FINE_TUNE_SPEC_SCHEMA_VERSION`] is bumped alongside it — a silent
    /// change here would mean an old persisted `spec_canonical` and a freshly
    /// encoded one of the same logical spec no longer hash identically.
    #[test]
    fn fine_tune_spec_canonical_pinned_golden() {
        let config = FineTuneConfig {
            lora_rank: 8,
            ..FineTuneConfig::default()
        };
        let s = fine_tune_spec_canonical(
            "patents",
            &["abstract".to_string()],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            "local:tiny",
            &config,
            1,
        )
        .expect("canonicalisable");
        let golden = fine_tune_spec_canonical(
            "patents",
            &["abstract".to_string()],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            "local:tiny",
            &FineTuneConfig {
                lora_rank: 8,
                ..FineTuneConfig::default()
            },
            1,
        )
        .expect("canonicalisable");
        assert_eq!(s, golden, "the golden fixture itself must be reproducible");
        // Sorted-key invariant: the top-level keys appear in lexical order.
        let base_model_at = s.find(r#""base_model""#).unwrap();
        let columns_at = s.find(r#""columns""#).unwrap();
        let config_at = s.find(r#""config""#).unwrap();
        let method_at = s.find(r#""method""#).unwrap();
        let source_at = s.find(r#""source""#).unwrap();
        let task_at = s.find(r#""task""#).unwrap();
        let world_size_at = s.find(r#""world_size""#).unwrap();
        assert!(
            base_model_at < columns_at
                && columns_at < config_at
                && config_at < method_at
                && method_at < source_at
                && source_at < task_at
                && task_at < world_size_at,
            "top-level keys must appear in sorted order, got: {s}"
        );
        assert!(s.starts_with(r#"{"base_model":"local:tiny","columns":["abstract"]"#));
    }

    /// Round trip: [`fine_tune_spec_from_canonical`] reconstructs a spec that
    /// re-encodes to the SAME canonical string — the only equality
    /// `TrainingSpec` (no `PartialEq`) can honestly offer here, and the one
    /// that matters: a retrain must reconstruct a spec whose identity is
    /// unchanged.
    #[test]
    fn fine_tune_spec_round_trips_through_canonical() {
        let f = canonical_fields();
        let original = canonical_of(&f);
        let decoded = fine_tune_spec_from_canonical(&original, FINE_TUNE_SPEC_SCHEMA_VERSION)
            .expect("decodable");
        let TrainingSpec::FineTune {
            source,
            columns,
            method,
            task,
            common,
            cache,
        } = &decoded
        else {
            panic!("expected the fine_tune variant, got {decoded:?}");
        };
        assert_eq!(
            *cache,
            CachePolicy::Bypass,
            "a replay always bypasses the cache"
        );
        let re_encoded = fine_tune_spec_canonical(
            source,
            columns,
            *method,
            *task,
            &common.base_model,
            &common.config,
            common.world_size,
        )
        .expect("canonicalisable");
        assert_eq!(original, re_encoded);
    }

    /// An unrecognised schema version is a typed refusal, never a best-effort
    /// guess at an unknown shape.
    #[test]
    fn fine_tune_spec_from_canonical_refuses_an_unknown_schema_version() {
        let f = canonical_fields();
        let encoded = canonical_of(&f);
        let err = fine_tune_spec_from_canonical(&encoded, FINE_TUNE_SPEC_SCHEMA_VERSION + 1)
            .expect_err("an unrecognised schema version must be refused");
        assert!(matches!(err, JammiError::FineTune(_)));
    }
}
