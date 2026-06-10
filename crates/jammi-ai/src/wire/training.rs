//! `TrainingService` spec-oneof ↔ engine `TrainingSpec` conversions.
//!
//! The transport-neutral `FineTuneConfig` / method conversions live on the wire
//! substrate ([`jammi_wire`]); what stays here are the conversions that touch the
//! engine spec vocabulary (`TrainingSpec`, the graph sampler, the
//! context-predictor config), which is only reachable in a `local` build.
//!
//! The `StartTraining` spec `oneof` mirrors the engine's [`TrainingSpec`] enum
//! variant-for-variant, field-for-field: a decoded request reconstructs the
//! identical engine spec, so a remote-submitted job is byte-identical to one
//! submitted in-process. Validation stays in the engine (the submit verbs call
//! `validate`); this is a pure shape map.

use tonic::Status;

use crate::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources, GraphSampleConfig};
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
use crate::fine_tune::FineTuneConfig;
use crate::pipeline::context_predictor::{
    ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
};

use jammi_wire::proto::training as pb;
use jammi_wire::{
    config_to_proto, method_from_proto, method_to_proto, model_task_from_proto, model_task_to_proto,
};

// ─── StartTraining spec oneof ↔ engine TrainingSpec ──────────────────────────
//
// The `oneof` carries the verb that produced the job; decode reconstructs the
// engine [`TrainingSpec`] field-for-field so a worker re-runs the identical job.
// The two LoRA fine-tune kinds carry their base-model + config in the request's
// common `base_model`/`config` fields (folded into [`TrainingCommon`]); the
// context-predictor kind carries its full budget inside `predictor_spec`.

/// Decode a [`pb::StartTrainingRequest`] into the engine [`TrainingSpec`]. The
/// `oneof` selects the variant; `base_model`/`config` fold into the two LoRA
/// kinds' [`TrainingCommon`]. A request with no spec set is malformed.
pub fn training_spec_from_proto(req: pb::StartTrainingRequest) -> Result<TrainingSpec, Status> {
    let pb::StartTrainingRequest {
        spec,
        base_model,
        config,
    } = req;
    let spec =
        spec.ok_or_else(|| Status::invalid_argument("StartTraining request carries no spec"))?;
    match spec {
        pb::start_training_request::Spec::FineTune(ft) => {
            let common = lora_common_from_proto(base_model, config)?;
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
        pb::start_training_request::Spec::GraphFineTune(g) => {
            let common = lora_common_from_proto(base_model, config)?;
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
        pb::start_training_request::Spec::ContextPredictor(cp) => {
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
/// LoRA kinds carry) onto a [`pb::StartTrainingRequest`] — the inverse of
/// [`training_spec_from_proto`], for the remote send side. The context-predictor
/// kind ignores `base_model`/`config` (its budget rides in `predictor_spec`), so
/// they are left empty there.
pub fn training_spec_to_proto(spec: &TrainingSpec) -> pb::StartTrainingRequest {
    match spec {
        TrainingSpec::FineTune {
            source,
            columns,
            method,
            task,
            common,
        } => pb::StartTrainingRequest {
            spec: Some(pb::start_training_request::Spec::FineTune(
                pb::FineTuneSpec {
                    source: source.clone(),
                    columns: columns.clone(),
                    method: method_to_proto(*method) as i32,
                    task: model_task_to_proto(*task) as i32,
                },
            )),
            base_model: common.base_model.clone(),
            config: Some(config_to_proto(&common.config)),
        },
        TrainingSpec::GraphFineTune {
            sources,
            sample_config,
            common,
        } => pb::StartTrainingRequest {
            spec: Some(pb::start_training_request::Spec::GraphFineTune(
                pb::GraphFineTuneSpec {
                    sources: Some(graph_sources_to_proto(sources)),
                    sample_config: Some(graph_sample_config_to_proto(sample_config)),
                },
            )),
            base_model: common.base_model.clone(),
            config: Some(config_to_proto(&common.config)),
        },
        TrainingSpec::ContextPredictor {
            source,
            predictor_spec,
        } => pb::StartTrainingRequest {
            spec: Some(pb::start_training_request::Spec::ContextPredictor(
                pb::ContextPredictorSpec {
                    source: source.clone(),
                    predictor_spec: Some(predictor_config_to_proto(predictor_spec)),
                },
            )),
            base_model: String::new(),
            config: None,
        },
    }
}

/// Fold the request's common `base_model` + optional `config` into a
/// [`TrainingCommon`] for the two LoRA fine-tune kinds. An empty base model is a
/// client error (the worker has nothing to adapt).
fn lora_common_from_proto(
    base_model: String,
    config: Option<pb::FineTuneConfig>,
) -> Result<TrainingCommon, Status> {
    if base_model.is_empty() {
        return Err(Status::invalid_argument("base_model is required"));
    }
    let config = config
        .map(FineTuneConfig::try_from)
        .transpose()?
        .unwrap_or_default();
    Ok(TrainingCommon { base_model, config })
}

fn graph_sources_from_proto(s: pb::GraphFineTuneSources) -> Result<GraphFineTuneSources, Status> {
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

fn graph_sources_to_proto(s: &GraphFineTuneSources) -> pb::GraphFineTuneSources {
    pb::GraphFineTuneSources {
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
    match pb::EdgeProvenance::try_from(p) {
        Ok(pb::EdgeProvenance::Declared) => Ok(EdgeProvenance::Declared),
        Ok(pb::EdgeProvenance::Similarity) => Ok(EdgeProvenance::Similarity),
        Ok(pb::EdgeProvenance::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "edge provenance must be DECLARED or SIMILARITY",
        )),
    }
}

fn edge_provenance_to_proto(p: EdgeProvenance) -> pb::EdgeProvenance {
    match p {
        EdgeProvenance::Declared => pb::EdgeProvenance::Declared,
        EdgeProvenance::Similarity => pb::EdgeProvenance::Similarity,
    }
}

fn graph_sample_config_from_proto(c: pb::GraphSampleConfig) -> GraphSampleConfig {
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

fn graph_sample_config_to_proto(c: &GraphSampleConfig) -> pb::GraphSampleConfig {
    pb::GraphSampleConfig {
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
    c: pb::ContextPredictorTrainConfig,
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

fn predictor_config_to_proto(c: &ContextPredictorTrainConfig) -> pb::ContextPredictorTrainConfig {
    pb::ContextPredictorTrainConfig {
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
    match pb::ContextArchitecture::try_from(a) {
        Ok(pb::ContextArchitecture::Cnp) => Ok(ContextArchitecture::Cnp),
        Ok(pb::ContextArchitecture::AttnCnp) => Ok(ContextArchitecture::AttnCnp),
        Ok(pb::ContextArchitecture::Tnp) => Ok(ContextArchitecture::Tnp),
        Ok(pb::ContextArchitecture::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "context predictor architecture must be CNP, ATTN_CNP, or TNP",
        )),
    }
}

fn context_architecture_to_proto(a: ContextArchitecture) -> pb::ContextArchitecture {
    match a {
        ContextArchitecture::Cnp => pb::ContextArchitecture::Cnp,
        ContextArchitecture::AttnCnp => pb::ContextArchitecture::AttnCnp,
        ContextArchitecture::Tnp => pb::ContextArchitecture::Tnp,
    }
}

fn predictive_head_from_proto(h: pb::PredictiveHead) -> Result<PredictiveHead, Status> {
    use pb::predictive_head::Head;
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

fn predictive_head_to_proto(h: &PredictiveHead) -> pb::PredictiveHead {
    use pb::predictive_head::Head;
    let inner = match h {
        PredictiveHead::Gaussian { objective } => Head::Gaussian(pb::predictive_head::Gaussian {
            objective: Some(gaussian_objective_to_proto(*objective)),
        }),
        PredictiveHead::Quantile { levels } => Head::Quantile(pb::predictive_head::Quantile {
            levels: levels.clone(),
        }),
    };
    pb::PredictiveHead { head: Some(inner) }
}

fn gaussian_objective_from_proto(o: pb::GaussianObjective) -> Result<GaussianObjective, Status> {
    use pb::gaussian_objective::Objective;
    match o.objective {
        Some(Objective::Nll(n)) => Ok(GaussianObjective::Nll { beta: n.beta }),
        Some(Objective::Crps(_)) => Ok(GaussianObjective::Crps),
        None => Err(Status::invalid_argument(
            "gaussian objective carries no nll or crps variant",
        )),
    }
}

fn gaussian_objective_to_proto(o: GaussianObjective) -> pb::GaussianObjective {
    use pb::gaussian_objective::Objective;
    let inner = match o {
        GaussianObjective::Nll { beta } => Objective::Nll(pb::gaussian_objective::Nll { beta }),
        GaussianObjective::Crps => Objective::Crps(pb::gaussian_objective::Crps {}),
    };
    pb::GaussianObjective {
        objective: Some(inner),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-encode a [`pb::StartTrainingRequest`] into the spec it decoded from —
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
}
