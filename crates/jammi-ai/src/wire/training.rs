//! `JobService.SubmitJob` spec-oneof ↔ engine `TrainingSpec` conversions.
//!
//! The transport-neutral `FineTuneConfig` / method conversions live on the wire
//! substrate ([`jammi_wire`]); what stays here are the conversions that touch the
//! engine spec vocabulary (`TrainingSpec`, the graph sampler, the
//! context-predictor config), which is only reachable in a `local` build.
//!
//! The `SubmitJob` spec `oneof` mirrors the engine's [`TrainingSpec`] enum
//! variant-for-variant, field-for-field: a decoded request reconstructs the
//! identical engine spec, so a remote-submitted job is byte-identical to one
//! submitted in-process. Validation stays in the engine (the submit verbs call
//! `validate`); this is a pure shape map.

use jammi_db::error::JammiError;
use prost::Message;
use tonic::Status;

use crate::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources, GraphSampleConfig};
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
use crate::fine_tune::FineTuneConfig;
use crate::pipeline::context_predictor::{
    ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
};

use jammi_wire::proto::job as pb;
use jammi_wire::proto::training as training_pb;
use jammi_wire::{
    config_to_proto, method_from_proto, method_to_proto, model_task_from_proto, model_task_to_proto,
};

// ─── SubmitJob spec oneof ↔ engine TrainingSpec ──────────────────────────────
//
// The `oneof` carries the verb that produced the job; decode reconstructs the
// engine [`TrainingSpec`] field-for-field so a worker re-runs the identical job.
// The two LoRA fine-tune kinds carry their base-model + config in the request's
// common `base_model`/`config` fields (folded into [`TrainingCommon`]); the
// context-predictor kind carries its full budget inside `predictor_spec`.

/// Decode a serialized [`pb::SubmitJobRequest`] body into the engine
/// [`TrainingSpec`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote submit paths decode through one
/// shared seam ([`training_spec_from_proto`]). A body that is not a valid
/// `SubmitJobRequest` is a client error (`InvalidArgument`), matching how a
/// malformed spec is rejected.
pub fn training_spec_from_bytes(body: &[u8]) -> Result<TrainingSpec, Status> {
    let req = pb::SubmitJobRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed SubmitJob request: {e}")))?;
    training_spec_from_proto(req)
}

/// Decode a [`pb::SubmitJobRequest`] into the engine [`TrainingSpec`]. The
/// `oneof` selects the variant; `base_model`/`config` fold into the two LoRA
/// kinds' [`TrainingCommon`]. A request with no spec set is malformed.
pub fn training_spec_from_proto(req: pb::SubmitJobRequest) -> Result<TrainingSpec, Status> {
    let pb::SubmitJobRequest {
        spec,
        base_model,
        config,
        idempotency_key: _,
        world_size,
        cache,
    } = req;
    let spec = spec.ok_or_else(|| Status::invalid_argument("SubmitJob request carries no spec"))?;
    match spec {
        pb::submit_job_request::Spec::FineTune(ft) => {
            let common = lora_common_from_proto(
                base_model,
                config,
                world_size,
                cache,
                LoraSpecKind::FineTune,
            )?;
            if ft.source.is_empty() {
                return Err(Status::invalid_argument("source is required"));
            }
            // A column-source fine-tune with no columns has no training data to
            // detect a format from — a client error, rejected at decode rather
            // than deferred to a failing worker.
            if ft.columns.is_empty() {
                return Err(Status::invalid_argument("columns is required"));
            }
            Ok(TrainingSpec::FineTune {
                source: ft.source,
                columns: ft.columns,
                method: method_from_proto(ft.method)?,
                task: model_task_from_proto(ft.task)?,
                common,
            })
        }
        pb::submit_job_request::Spec::GraphFineTune(g) => {
            let common = lora_common_from_proto(
                base_model,
                config,
                world_size,
                cache,
                LoraSpecKind::GraphFineTune,
            )?;
            let sources = g.sources.ok_or_else(|| {
                Status::invalid_argument("graph_fine_tune spec carries no sources")
            })?;
            let sample_config = g.sample_config.ok_or_else(|| {
                Status::invalid_argument("graph_fine_tune spec carries no sample_config")
            })?;
            Ok(TrainingSpec::GraphFineTune {
                sources: graph_sources_from_proto(sources)?,
                sample_config: graph_sample_config_from_proto(sample_config),
                common,
            })
        }
        pb::submit_job_request::Spec::ContextPredictor(cp) => {
            // The context-predictor kind carries no `TrainingCommon`, so the
            // engine spec has nowhere to put a rank count and a multi-rank
            // predictor job is unrepresentable past this point. That makes
            // this decode the LAST edge that can still see the count a caller
            // chose: refuse here, typed, rather than silently drop it and run
            // the single-rank job the caller did not ask for. The wire's `0`
            // (unset) and `1` are both the single rank and pass.
            if world_size > 1 {
                let message = format!(
                    "world_size = {world_size} is not supported for a context_predictor job \
                     (the episodic meta-training loop runs on a single rank); submit it \
                     without a rank count or with world_size = 1"
                );
                let engine_err = JammiError::Config(message.clone());
                return Err(jammi_wire::attach_error_detail(
                    tonic::Code::InvalidArgument,
                    message,
                    &engine_err,
                ));
            }
            let predictor_spec = cp.predictor_spec.ok_or_else(|| {
                Status::invalid_argument("context_predictor spec carries no predictor_spec")
            })?;
            Ok(TrainingSpec::ContextPredictor {
                source: cp.source,
                predictor_spec: predictor_config_from_proto(predictor_spec)?,
            })
        }
    }
}

/// Encode the engine [`TrainingSpec`] (plus the common base-model + config the
/// LoRA kinds carry) onto a [`pb::SubmitJobRequest`] — the inverse of
/// [`training_spec_from_proto`], for the remote send side. The context-predictor
/// kind ignores `base_model`/`config` (its budget rides in `predictor_spec`), so
/// they are left empty there.
pub fn training_spec_to_proto(spec: &TrainingSpec) -> pb::SubmitJobRequest {
    match spec {
        TrainingSpec::FineTune {
            source,
            columns,
            method,
            task,
            common,
        } => pb::SubmitJobRequest {
            spec: Some(pb::submit_job_request::Spec::FineTune(
                training_pb::FineTuneSpec {
                    source: source.clone(),
                    columns: columns.clone(),
                    method: method_to_proto(*method) as i32,
                    task: model_task_to_proto(*task) as i32,
                },
            )),
            base_model: common.base_model.clone(),
            config: Some(config_to_proto(&common.config)),
            idempotency_key: String::new(),
            world_size: common.world_size,
            cache: super::cache::cache_policy_to_proto(common.cache) as i32,
        },
        TrainingSpec::GraphFineTune {
            sources,
            sample_config,
            common,
        } => pb::SubmitJobRequest {
            spec: Some(pb::submit_job_request::Spec::GraphFineTune(
                training_pb::GraphFineTuneSpec {
                    sources: Some(graph_sources_to_proto(sources)),
                    sample_config: Some(graph_sample_config_to_proto(sample_config)),
                },
            )),
            base_model: common.base_model.clone(),
            config: Some(config_to_proto(&common.config)),
            idempotency_key: String::new(),
            world_size: common.world_size,
            cache: super::cache::cache_policy_to_proto(common.cache) as i32,
        },
        TrainingSpec::ContextPredictor {
            source,
            predictor_spec,
        } => pb::SubmitJobRequest {
            spec: Some(pb::submit_job_request::Spec::ContextPredictor(
                training_pb::ContextPredictorSpec {
                    source: source.clone(),
                    predictor_spec: Some(predictor_config_to_proto(predictor_spec)),
                },
            )),
            base_model: String::new(),
            config: None,
            idempotency_key: String::new(),
            // The predictor kind has no rank count to carry: `0` is the
            // wire's unset value, and the decode above refuses anything
            // greater than one rank for this spec.
            world_size: 0,
            // The predictor kind has no `TrainingCommon`, so it has nothing
            // to probe a cache hit by; `UNSPECIFIED` decodes to the engine's
            // `Bypass` default the same way an unset `world_size` decodes to
            // one rank.
            cache: 0,
        },
    }
}

/// Which of the two LoRA fine-tune kinds is decoding through
/// [`lora_common_from_proto`] — the ONE place that can still see which kind
/// requested a policy it cannot honour (P4, U3 fix round 1), the same "last
/// edge that can still see it" reasoning [`training_spec_from_proto`]'s own
/// `ContextPredictor` world_size refusal already uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoraSpecKind {
    FineTune,
    GraphFineTune,
}

/// Fold the request's common `base_model` + optional `config` + `cache` into a
/// [`TrainingCommon`] for the two LoRA fine-tune kinds. An empty base model is a
/// client error (the worker has nothing to adapt).
///
/// P4 (fix round 1): `cache = USE` is refused, typed, for `kind ==
/// GraphFineTune` — `ProducingDescriptor::FineTune` (and every
/// `probe_model_by_definition`/`record_model_materialization` mechanism
/// built on it) covers only the column-source `FineTune` kind at this
/// commit (`worker.rs`'s own `materialization_source: None` for the graph
/// path); a graph fine-tune job carries no materialization to probe or
/// record, so honouring `Use` for it would be a silent no-op behind a wire
/// promise the engine cannot keep. This mirrors the `ContextPredictor`
/// `world_size` refusal above: the policy the kind cannot honour is refused
/// at the LAST edge that can still see both the kind and the value, rather
/// than silently dropped once the two are folded into one `TrainingCommon`.
/// `Bypass`/unset is unaffected — a graph fine-tune always trains, exactly
/// as it did before this field existed.
fn lora_common_from_proto(
    base_model: String,
    config: Option<training_pb::FineTuneConfig>,
    world_size: u32,
    cache: i32,
    kind: LoraSpecKind,
) -> Result<TrainingCommon, Status> {
    if base_model.is_empty() {
        return Err(Status::invalid_argument("base_model is required"));
    }
    let config = config
        .map(FineTuneConfig::try_from)
        .transpose()?
        .unwrap_or_default();
    let cache = super::cache_policy_from_proto(cache)?;
    if kind == LoraSpecKind::GraphFineTune && cache == jammi_db::store::CachePolicy::Use {
        let message = "cache = USE is not supported for a graph_fine_tune job (a graph \
                        fine-tune carries no `ProducingDescriptor::FineTune` materialization to \
                        probe or record — that descriptor covers only the column-source \
                        `fine_tune` kind); submit it without `cache` or with `cache = BYPASS`"
            .to_string();
        let engine_err = JammiError::Config(message.clone());
        return Err(jammi_wire::attach_error_detail(
            tonic::Code::InvalidArgument,
            message,
            &engine_err,
        ));
    }
    Ok(TrainingCommon {
        base_model,
        config,
        world_size: world_size_from_proto(world_size),
        cache,
    })
}

/// Resolve the wire's rank count into the engine's.
///
/// `world_size` is an implicit-presence `uint32`, so `0` is what a request
/// that never set the field carries — indistinguishable from one encoded
/// without the field at all. It resolves to
/// [`crate::fine_tune::spec::DEFAULT_WORLD_SIZE`] HERE, at the decode, so the
/// wire's unset value never reaches a persisted spec: a `0` written into
/// `jobs.spec` would be a zero-rank job on disk, which no worker can place.
/// Every other count passes through unchanged and is bounded at the submit
/// edge against what the deployment can serve — this seam resolves, it never
/// refuses a count for being too large.
fn world_size_from_proto(world_size: u32) -> u32 {
    if world_size == 0 {
        crate::fine_tune::spec::DEFAULT_WORLD_SIZE
    } else {
        world_size
    }
}

fn graph_sources_from_proto(
    s: training_pb::GraphFineTuneSources,
) -> Result<GraphFineTuneSources, Status> {
    Ok(GraphFineTuneSources {
        node_source: s.node_source,
        id_column: s.id_column,
        text_column: s.text_column,
        edge_source: s.edge_source,
        src_column: s.src_column,
        dst_column: s.dst_column,
        provenance: edge_provenance_from_proto(s.provenance)?,
    })
}

fn graph_sources_to_proto(s: &GraphFineTuneSources) -> training_pb::GraphFineTuneSources {
    training_pb::GraphFineTuneSources {
        node_source: s.node_source.clone(),
        id_column: s.id_column.clone(),
        text_column: s.text_column.clone(),
        edge_source: s.edge_source.clone(),
        src_column: s.src_column.clone(),
        dst_column: s.dst_column.clone(),
        provenance: edge_provenance_to_proto(s.provenance) as i32,
    }
}

fn edge_provenance_from_proto(p: i32) -> Result<EdgeProvenance, Status> {
    match training_pb::EdgeProvenance::try_from(p) {
        Ok(training_pb::EdgeProvenance::Declared) => Ok(EdgeProvenance::Declared),
        Ok(training_pb::EdgeProvenance::Similarity) => Ok(EdgeProvenance::Similarity),
        Ok(training_pb::EdgeProvenance::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "edge provenance must be DECLARED or SIMILARITY",
        )),
    }
}

fn edge_provenance_to_proto(p: EdgeProvenance) -> training_pb::EdgeProvenance {
    match p {
        EdgeProvenance::Declared => training_pb::EdgeProvenance::Declared,
        EdgeProvenance::Similarity => training_pb::EdgeProvenance::Similarity,
    }
}

fn graph_sample_config_from_proto(c: training_pb::GraphSampleConfig) -> GraphSampleConfig {
    GraphSampleConfig {
        walk_length: c.walk_length as usize,
        walks_per_node: c.walks_per_node as usize,
        return_p: c.return_p,
        in_out_q: c.in_out_q,
        hard_negatives: c.hard_negatives as usize,
        exclude_hops: c.exclude_hops as usize,
        min_negatives: c.min_negatives as usize,
        seed: c.seed,
    }
}

fn graph_sample_config_to_proto(c: &GraphSampleConfig) -> training_pb::GraphSampleConfig {
    training_pb::GraphSampleConfig {
        walk_length: c.walk_length as u32,
        walks_per_node: c.walks_per_node as u32,
        return_p: c.return_p,
        in_out_q: c.in_out_q,
        hard_negatives: c.hard_negatives as u32,
        exclude_hops: c.exclude_hops as u32,
        min_negatives: c.min_negatives as u32,
        seed: c.seed,
    }
}

fn predictor_config_from_proto(
    c: training_pb::ContextPredictorTrainConfig,
) -> Result<ContextPredictorTrainConfig, Status> {
    let head = c
        .head
        .ok_or_else(|| Status::invalid_argument("context predictor spec carries no head"))?;
    Ok(ContextPredictorTrainConfig {
        model_id: c.model_id,
        architecture: context_architecture_from_proto(c.architecture)?,
        key_column: c.key_column,
        task_column: c.task_column,
        value_column: c.value_column,
        context_k: c.context_k as usize,
        hidden_dim: c.hidden_dim as usize,
        num_heads: c.num_heads as usize,
        num_layers: c.num_layers as usize,
        head: predictive_head_from_proto(head)?,
        epochs: c.epochs as usize,
        learning_rate: c.learning_rate,
        grad_clip: c.grad_clip,
        test_task_fraction: c.test_task_fraction,
        min_task_count: c.min_task_count as usize,
        seed: c.seed,
    })
}

fn predictor_config_to_proto(
    c: &ContextPredictorTrainConfig,
) -> training_pb::ContextPredictorTrainConfig {
    training_pb::ContextPredictorTrainConfig {
        model_id: c.model_id.clone(),
        architecture: context_architecture_to_proto(c.architecture) as i32,
        key_column: c.key_column.clone(),
        task_column: c.task_column.clone(),
        value_column: c.value_column.clone(),
        context_k: c.context_k as u32,
        hidden_dim: c.hidden_dim as u32,
        num_heads: c.num_heads as u32,
        num_layers: c.num_layers as u32,
        head: Some(predictive_head_to_proto(&c.head)),
        epochs: c.epochs as u32,
        learning_rate: c.learning_rate,
        grad_clip: c.grad_clip,
        test_task_fraction: c.test_task_fraction,
        min_task_count: c.min_task_count as u32,
        seed: c.seed,
    }
}

fn context_architecture_from_proto(a: i32) -> Result<ContextArchitecture, Status> {
    match training_pb::ContextArchitecture::try_from(a) {
        Ok(training_pb::ContextArchitecture::Cnp) => Ok(ContextArchitecture::Cnp),
        Ok(training_pb::ContextArchitecture::AttnCnp) => Ok(ContextArchitecture::AttnCnp),
        Ok(training_pb::ContextArchitecture::Tnp) => Ok(ContextArchitecture::Tnp),
        Ok(training_pb::ContextArchitecture::Unspecified) | Err(_) => {
            Err(Status::invalid_argument(
                "context predictor architecture must be CNP, ATTN_CNP, or TNP",
            ))
        }
    }
}

fn context_architecture_to_proto(a: ContextArchitecture) -> training_pb::ContextArchitecture {
    match a {
        ContextArchitecture::Cnp => training_pb::ContextArchitecture::Cnp,
        ContextArchitecture::AttnCnp => training_pb::ContextArchitecture::AttnCnp,
        ContextArchitecture::Tnp => training_pb::ContextArchitecture::Tnp,
    }
}

fn predictive_head_from_proto(h: training_pb::PredictiveHead) -> Result<PredictiveHead, Status> {
    use training_pb::predictive_head::Head;
    match h.head {
        Some(Head::Gaussian(g)) => {
            let objective = g.objective.ok_or_else(|| {
                Status::invalid_argument("gaussian predictive head carries no objective")
            })?;
            Ok(PredictiveHead::Gaussian {
                objective: gaussian_objective_from_proto(objective)?,
            })
        }
        Some(Head::Quantile(q)) => Ok(PredictiveHead::Quantile { levels: q.levels }),
        None => Err(Status::invalid_argument(
            "predictive head carries no gaussian or quantile variant",
        )),
    }
}

fn predictive_head_to_proto(h: &PredictiveHead) -> training_pb::PredictiveHead {
    use training_pb::predictive_head::Head;
    let inner = match h {
        PredictiveHead::Gaussian { objective } => {
            Head::Gaussian(training_pb::predictive_head::Gaussian {
                objective: Some(gaussian_objective_to_proto(*objective)),
            })
        }
        PredictiveHead::Quantile { levels } => {
            Head::Quantile(training_pb::predictive_head::Quantile {
                levels: levels.clone(),
            })
        }
    };
    training_pb::PredictiveHead { head: Some(inner) }
}

fn gaussian_objective_from_proto(
    o: training_pb::GaussianObjective,
) -> Result<GaussianObjective, Status> {
    use training_pb::gaussian_objective::Objective;
    match o.objective {
        Some(Objective::Nll(n)) => Ok(GaussianObjective::Nll { beta: n.beta }),
        Some(Objective::Crps(_)) => Ok(GaussianObjective::Crps),
        None => Err(Status::invalid_argument(
            "gaussian objective carries no nll or crps variant",
        )),
    }
}

fn gaussian_objective_to_proto(o: GaussianObjective) -> training_pb::GaussianObjective {
    use training_pb::gaussian_objective::Objective;
    let inner = match o {
        GaussianObjective::Nll { beta } => {
            Objective::Nll(training_pb::gaussian_objective::Nll { beta })
        }
        GaussianObjective::Crps => Objective::Crps(training_pb::gaussian_objective::Crps {}),
    };
    training_pb::GaussianObjective {
        objective: Some(inner),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-encode a [`pb::SubmitJobRequest`] into the spec it decoded from —
    /// `training_spec_to_proto` then `training_spec_from_proto` — so a remote
    /// `GraphFineTune` job is byte-identical to the in-process one. Every field
    /// of the source spec is a distinctive non-default value, and every field of
    /// the decoded spec is asserted individually: a dropped or mis-mapped field
    /// in either direction fails the test.
    #[test]
    fn graph_fine_tune_spec_round_trips_field_for_field() {
        let original = TrainingSpec::GraphFineTune {
            sources: GraphFineTuneSources {
                node_source: "nodes_src".into(),
                id_column: "node_id".into(),
                text_column: "node_text".into(),
                edge_source: "edges_src".into(),
                src_column: "edge_from".into(),
                dst_column: "edge_to".into(),
                provenance: EdgeProvenance::Similarity,
            },
            sample_config: GraphSampleConfig {
                walk_length: 7,
                walks_per_node: 3,
                return_p: 0.25,
                in_out_q: 4.0,
                hard_negatives: 5,
                exclude_hops: 2,
                min_negatives: 9,
                seed: 0xDEAD_BEEF,
            },
            common: TrainingCommon {
                base_model: "graph-base".into(),
                // A non-default config knob so the common config round-trips too.
                config: FineTuneConfig {
                    lora_rank: 32,
                    ..FineTuneConfig::default()
                },
                world_size: crate::fine_tune::spec::DEFAULT_WORLD_SIZE,
                cache: jammi_db::store::CachePolicy::Bypass,
            },
        };

        let proto = training_spec_to_proto(&original);
        let decoded =
            training_spec_from_proto(proto).expect("graph spec round-trips through decode");

        let TrainingSpec::GraphFineTune {
            sources,
            sample_config,
            common,
        } = decoded
        else {
            panic!("decoded spec is not GraphFineTune");
        };

        assert_eq!(sources.node_source, "nodes_src");
        assert_eq!(sources.id_column, "node_id");
        assert_eq!(sources.text_column, "node_text");
        assert_eq!(sources.edge_source, "edges_src");
        assert_eq!(sources.src_column, "edge_from");
        assert_eq!(sources.dst_column, "edge_to");
        assert_eq!(sources.provenance, EdgeProvenance::Similarity);

        assert_eq!(sample_config.walk_length, 7);
        assert_eq!(sample_config.walks_per_node, 3);
        assert_eq!(sample_config.return_p, 0.25);
        assert_eq!(sample_config.in_out_q, 4.0);
        assert_eq!(sample_config.hard_negatives, 5);
        assert_eq!(sample_config.exclude_hops, 2);
        assert_eq!(sample_config.min_negatives, 9);
        assert_eq!(sample_config.seed, 0xDEAD_BEEF);

        assert_eq!(common.base_model, "graph-base");
        assert_eq!(common.config.lora_rank, 32);
    }

    /// The `ContextPredictor` spec round-trips field-for-field through
    /// `training_spec_to_proto` → `training_spec_from_proto`, with the Gaussian
    /// `Nll { beta }` head. Every scalar / column / budget knob is a distinctive
    /// non-default value asserted individually.
    #[test]
    fn context_predictor_spec_round_trips_gaussian_head() {
        let original = TrainingSpec::ContextPredictor {
            source: "episodes_src".into(),
            predictor_spec: ContextPredictorTrainConfig {
                model_id: "ctx-pred-1".into(),
                architecture: ContextArchitecture::Tnp,
                key_column: "row_key".into(),
                task_column: "cohort".into(),
                value_column: "outcome".into(),
                context_k: 13,
                hidden_dim: 256,
                num_heads: 8,
                num_layers: 6,
                head: PredictiveHead::Gaussian {
                    objective: GaussianObjective::Nll { beta: 0.5 },
                },
                epochs: 11,
                learning_rate: 3e-4,
                grad_clip: 2.5,
                test_task_fraction: 0.3,
                min_task_count: 7,
                seed: 0xC0FF_EE42,
            },
        };

        let proto = training_spec_to_proto(&original);
        let decoded =
            training_spec_from_proto(proto).expect("predictor spec round-trips through decode");

        let TrainingSpec::ContextPredictor {
            source,
            predictor_spec,
        } = decoded
        else {
            panic!("decoded spec is not ContextPredictor");
        };

        assert_eq!(source, "episodes_src");
        assert_eq!(predictor_spec.model_id, "ctx-pred-1");
        assert_eq!(predictor_spec.architecture, ContextArchitecture::Tnp);
        assert_eq!(predictor_spec.key_column, "row_key");
        assert_eq!(predictor_spec.task_column, "cohort");
        assert_eq!(predictor_spec.value_column, "outcome");
        assert_eq!(predictor_spec.context_k, 13);
        assert_eq!(predictor_spec.hidden_dim, 256);
        assert_eq!(predictor_spec.num_heads, 8);
        assert_eq!(predictor_spec.num_layers, 6);
        match predictor_spec.head {
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Nll { beta },
            } => assert_eq!(beta, 0.5),
            other => panic!("expected Gaussian Nll head, got {other:?}"),
        }
        assert_eq!(predictor_spec.epochs, 11);
        assert_eq!(predictor_spec.learning_rate, 3e-4);
        assert_eq!(predictor_spec.grad_clip, 2.5);
        assert_eq!(predictor_spec.test_task_fraction, 0.3);
        assert_eq!(predictor_spec.min_task_count, 7);
        assert_eq!(predictor_spec.seed, 0xC0FF_EE42);
    }

    /// The predictor spec's other two head shapes also round-trip: the Gaussian
    /// `Crps` objective (no payload) and the `Quantile { levels }` head (a
    /// distinctive non-default level vector).
    #[test]
    fn context_predictor_spec_round_trips_crps_and_quantile_heads() {
        for head in [
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Crps,
            },
            PredictiveHead::Quantile {
                levels: vec![0.1, 0.5, 0.9],
            },
        ] {
            let original = TrainingSpec::ContextPredictor {
                source: "episodes_src".into(),
                predictor_spec: ContextPredictorTrainConfig {
                    model_id: "ctx-pred-2".into(),
                    architecture: ContextArchitecture::AttnCnp,
                    key_column: "row_key".into(),
                    task_column: "cohort".into(),
                    value_column: "outcome".into(),
                    context_k: 4,
                    hidden_dim: 64,
                    num_heads: 4,
                    num_layers: 2,
                    head: head.clone(),
                    epochs: 2,
                    learning_rate: 1e-3,
                    grad_clip: 1.0,
                    test_task_fraction: 0.2,
                    min_task_count: 3,
                    seed: 42,
                },
            };

            let proto = training_spec_to_proto(&original);
            let decoded =
                training_spec_from_proto(proto).expect("predictor spec round-trips through decode");

            let TrainingSpec::ContextPredictor { predictor_spec, .. } = decoded else {
                panic!("decoded spec is not ContextPredictor");
            };

            match (&head, &predictor_spec.head) {
                (
                    PredictiveHead::Gaussian {
                        objective: GaussianObjective::Crps,
                    },
                    PredictiveHead::Gaussian {
                        objective: GaussianObjective::Crps,
                    },
                ) => {}
                (
                    PredictiveHead::Quantile { levels: want },
                    PredictiveHead::Quantile { levels: got },
                ) => assert_eq!(got, want),
                (want, got) => panic!("head mismatch: wanted {want:?}, got {got:?}"),
            }
        }
    }

    /// A minimal column-source request carrying `world_size`, so the count is
    /// the only thing these three tests vary.
    fn fine_tune_request(world_size: u32) -> pb::SubmitJobRequest {
        pb::SubmitJobRequest {
            spec: Some(pb::submit_job_request::Spec::FineTune(
                training_pb::FineTuneSpec {
                    source: "patents".into(),
                    columns: vec!["abstract".into()],
                    method: method_to_proto(crate::fine_tune::FineTuneMethod::Lora) as i32,
                    task: model_task_to_proto(jammi_db::ModelTask::TextEmbedding) as i32,
                },
            )),
            base_model: "local:tiny".into(),
            config: None,
            idempotency_key: String::new(),
            world_size,
            cache: 0,
        }
    }

    fn decoded_common(req: pb::SubmitJobRequest) -> TrainingCommon {
        match training_spec_from_proto(req).expect("decode") {
            TrainingSpec::FineTune { common, .. } => common,
            other => panic!("expected the fine_tune variant, got {other:?}"),
        }
    }

    /// The wire's unset count (`0`) resolves to the engine's single rank AT
    /// THE DECODE, before the spec can be persisted: a `0` reaching
    /// `jobs.spec` would be a zero-rank job on disk that no worker can place.
    /// A chosen count passes through unchanged.
    #[test]
    fn the_wire_unset_count_resolves_to_one_rank_and_a_chosen_count_survives() {
        assert_eq!(
            decoded_common(fine_tune_request(0)).world_size,
            1,
            "0 is the wire's UNSET value and must decode to one rank"
        );
        assert_eq!(decoded_common(fine_tune_request(1)).world_size, 1);
        assert_eq!(
            decoded_common(fine_tune_request(3)).world_size,
            3,
            "a chosen count is carried into the spec, never defaulted away"
        );
    }

    /// The count makes the round trip back onto the request, so a spec
    /// re-encoded for a remote send carries the rank count it was submitted
    /// with rather than silently reverting to one rank.
    #[test]
    fn the_rank_count_round_trips_back_onto_the_request() {
        let spec = training_spec_from_proto(fine_tune_request(2)).expect("decode");
        assert_eq!(training_spec_to_proto(&spec).world_size, 2);
    }

    /// [`fine_tune_request`], but with the `cache` field the only thing that
    /// varies — so `lora_common_from_proto`'s cache mapping is the only
    /// determinant under test.
    fn fine_tune_request_with_cache(cache: i32) -> pb::SubmitJobRequest {
        pb::SubmitJobRequest {
            cache,
            ..fine_tune_request(0)
        }
    }

    /// UNSPECIFIED (the wire's unset value) and the explicit `BYPASS` both
    /// decode to the engine's `CachePolicy::Bypass` — the documented mapping
    /// every other `*_UNSPECIFIED` arm in this crate follows, so an unset
    /// field costs a pre-existing remote caller nothing.
    #[test]
    fn unspecified_and_bypass_cache_both_decode_to_bypass() {
        use jammi_wire::proto::inference::CachePolicy as ProtoCachePolicy;

        assert_eq!(
            decoded_common(fine_tune_request_with_cache(
                ProtoCachePolicy::Unspecified as i32
            ))
            .cache,
            jammi_db::store::CachePolicy::Bypass
        );
        assert_eq!(
            decoded_common(fine_tune_request_with_cache(
                ProtoCachePolicy::Bypass as i32
            ))
            .cache,
            jammi_db::store::CachePolicy::Bypass
        );
    }

    /// `CACHE_POLICY_USE` decodes to the engine's `CachePolicy::Use` — the
    /// opt-in model-level cache reuse this field exists to reach.
    #[test]
    fn cache_use_decodes_to_the_engine_use_policy() {
        use jammi_wire::proto::inference::CachePolicy as ProtoCachePolicy;

        assert_eq!(
            decoded_common(fine_tune_request_with_cache(ProtoCachePolicy::Use as i32)).cache,
            jammi_db::store::CachePolicy::Use
        );
    }

    /// An out-of-range `cache` value is a loud client error, never a silent
    /// fall-through to the default — matching every other enum decode in this
    /// module.
    #[test]
    fn an_out_of_range_cache_value_is_a_loud_error() {
        let status = training_spec_from_proto(fine_tune_request_with_cache(99))
            .expect_err("an out-of-range cache value must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    /// The cache choice makes the round trip back onto the request, exactly
    /// like the rank count: a spec re-encoded for a remote send carries the
    /// cache policy it was submitted with rather than reverting to the
    /// default.
    #[test]
    fn the_cache_policy_round_trips_back_onto_the_request() {
        use jammi_wire::proto::inference::CachePolicy as ProtoCachePolicy;

        let spec =
            training_spec_from_proto(fine_tune_request_with_cache(ProtoCachePolicy::Use as i32))
                .expect("decode");
        assert_eq!(
            training_spec_to_proto(&spec).cache,
            ProtoCachePolicy::Use as i32
        );
    }

    /// The cache field rides on the request like `world_size`, not the kind
    /// — there is only one place to put it — but P4 (fix round 1) means the
    /// two LoRA kinds do NOT honour it identically: `cache = USE` on a
    /// `graph_fine_tune` job is refused, typed, at this same decode (a graph
    /// fine-tune carries no `ProducingDescriptor::FineTune` materialization
    /// to probe or record — `worker.rs`'s own `materialization_source: None`
    /// for this kind). This replaces the prior version of this test, which
    /// asserted the CARRY; asserting the refusal is the correctness bar now.
    #[test]
    fn graph_fine_tune_refuses_cache_use() {
        use jammi_wire::proto::inference::CachePolicy as ProtoCachePolicy;

        let request = pb::SubmitJobRequest {
            spec: Some(pb::submit_job_request::Spec::GraphFineTune(
                training_pb::GraphFineTuneSpec {
                    sources: Some(training_pb::GraphFineTuneSources {
                        node_source: "nodes".into(),
                        id_column: "id".into(),
                        text_column: "text".into(),
                        edge_source: "edges".into(),
                        src_column: "src".into(),
                        dst_column: "dst".into(),
                        provenance: training_pb::EdgeProvenance::Declared as i32,
                    }),
                    sample_config: Some(training_pb::GraphSampleConfig::default()),
                },
            )),
            base_model: "local:tiny".into(),
            config: None,
            idempotency_key: String::new(),
            world_size: 0,
            cache: ProtoCachePolicy::Use as i32,
        };

        let status = training_spec_from_proto(request)
            .expect_err("cache = USE must be refused for graph_fine_tune");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    /// The refusal above is scoped to `USE` specifically — `BYPASS`/unset
    /// (the default) still decodes and carries onto a graph fine-tune's
    /// `TrainingCommon` exactly as before this field existed: this kind is
    /// never blocked from training, only from a reuse promise it cannot
    /// keep.
    #[test]
    fn graph_fine_tune_still_carries_cache_bypass() {
        use jammi_wire::proto::inference::CachePolicy as ProtoCachePolicy;

        let request = pb::SubmitJobRequest {
            spec: Some(pb::submit_job_request::Spec::GraphFineTune(
                training_pb::GraphFineTuneSpec {
                    sources: Some(training_pb::GraphFineTuneSources {
                        node_source: "nodes".into(),
                        id_column: "id".into(),
                        text_column: "text".into(),
                        edge_source: "edges".into(),
                        src_column: "src".into(),
                        dst_column: "dst".into(),
                        provenance: training_pb::EdgeProvenance::Declared as i32,
                    }),
                    sample_config: Some(training_pb::GraphSampleConfig::default()),
                },
            )),
            base_model: "local:tiny".into(),
            config: None,
            idempotency_key: String::new(),
            world_size: 0,
            cache: ProtoCachePolicy::Bypass as i32,
        };

        let spec = training_spec_from_proto(request).expect("decode");
        let TrainingSpec::GraphFineTune { common, .. } = spec else {
            panic!("expected the graph_fine_tune variant");
        };
        assert_eq!(common.cache, jammi_db::store::CachePolicy::Bypass);
    }

    /// r26: a context-predictor job is refused above one rank at the LAST
    /// edge that can still see the count — the engine's
    /// `TrainingSpec::ContextPredictor` has no `TrainingCommon` and therefore
    /// no field to carry it, so a count that got past this decode would be
    /// silently dropped and the caller would get a single-rank job it never
    /// asked for. The unset and single-rank values still pass, and the
    /// re-encode leaves the wire's unset `0`.
    #[test]
    fn a_multi_rank_context_predictor_is_refused_at_the_decode() {
        // Encoded from a real engine spec, so the request under test is the
        // one the client sends rather than a hand-built shape that could
        // drift from it; only the count varies.
        let predictor_request = |world_size: u32| pb::SubmitJobRequest {
            world_size,
            ..training_spec_to_proto(&TrainingSpec::ContextPredictor {
                source: "episodes_src".into(),
                predictor_spec: ContextPredictorTrainConfig {
                    model_id: "ctx-pred-ranks".into(),
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
                },
            })
        };

        for unset_or_single in [0, 1] {
            training_spec_from_proto(predictor_request(unset_or_single))
                .expect("a single-rank context predictor is the job this kind has always been");
        }

        let status = training_spec_from_proto(predictor_request(2))
            .expect_err("a two-rank context predictor must be refused");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("context_predictor"),
            "the refusal must name the kind it refuses: {}",
            status.message()
        );
        let engine_err = jammi_wire::error_from_status(&status);
        assert!(
            matches!(engine_err, JammiError::Config(_)),
            "the refusal must reconstruct as the typed engine error, not the lossy \
             fallback: {engine_err:?}"
        );

        let spec = training_spec_from_proto(predictor_request(0)).expect("decode");
        assert_eq!(
            training_spec_to_proto(&spec).world_size,
            0,
            "the predictor kind carries no count, so the re-encode leaves the wire's unset value"
        );
    }
}
