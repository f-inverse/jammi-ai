//! `FineTuneConfig`'s exhaustive-destructuring completeness test.
//!
//! Every field of [`FineTuneConfig`] is named below — no `..` — so a field
//! appended to the struct fails THIS FILE to compile until it is bound (and
//! used) here too. This is the lever the descriptor's canonical-encoding
//! producer (`jammi-ai`'s `spec_canonical`) depends on: a
//! `FineTuneConfig` field that could silently escape a no-`..` destructuring
//! could silently escape the canonical encoding the fine-tune producer's
//! identity hash folds, two differently-configured runs colliding on one
//! hash. Proving the lever independently of that producer (rather than only
//! inside it) means a future field addition is caught here even before the
//! producer itself is touched.

use jammi_wire::fine_tune::{
    ClassificationLoss, ComputePrecision, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
    HardNegativeConfig, LoraInitMode, LrSchedule, RegressionLoss,
};

/// Every field of [`FineTuneConfig`] is bound below — no `..` — over the
/// engine default. Each binding is touched by an assertion against the
/// default value it carries, so the destructuring is real code (not dead,
/// unused-variable code clippy would flag) and a field whose default silently
/// changed would fail this test too.
#[test]
fn every_field_of_fine_tune_config_is_named_and_checked_against_the_default() {
    let cfg = FineTuneConfig::default();
    let FineTuneConfig {
        lora_rank,
        lora_alpha,
        lora_dropout,
        learning_rate,
        epochs,
        batch_size,
        max_seq_length,
        embedding_loss,
        classification_loss,
        regression_loss,
        quantile_levels,
        gradient_accumulation_steps,
        validation_fraction,
        early_stopping_patience,
        warmup_steps,
        lr_schedule,
        early_stopping_metric,
        target_modules,
        layers_to_transform,
        use_rslora,
        rank_pattern,
        init_lora_weights,
        backbone_dtype,
        weight_decay,
        max_grad_norm,
        cached,
        hard_negatives,
        matryoshka_dims,
        seed,
        keep_last_n_checkpoints,
    } = cfg;

    assert_eq!(lora_rank, 8);
    assert_eq!(lora_alpha, 16.0);
    assert_eq!(lora_dropout, 0.05);
    assert_eq!(learning_rate, 2e-4);
    assert_eq!(epochs, 3);
    assert_eq!(batch_size, 8);
    assert_eq!(max_seq_length, 512);
    assert_eq!(embedding_loss, None::<EmbeddingLoss>);
    assert_eq!(classification_loss, None::<ClassificationLoss>);
    assert_eq!(regression_loss, None::<RegressionLoss>);
    assert_eq!(quantile_levels, Vec::<f64>::new());
    assert_eq!(gradient_accumulation_steps, 1);
    assert_eq!(validation_fraction, 0.1);
    assert_eq!(early_stopping_patience, 3);
    assert_eq!(warmup_steps, 100);
    assert_eq!(lr_schedule, LrSchedule::CosineDecay);
    assert_eq!(early_stopping_metric, EarlyStoppingMetric::ValLoss);
    assert_eq!(target_modules, Vec::<String>::new());
    assert_eq!(layers_to_transform, None::<Vec<usize>>);
    assert!(!use_rslora);
    assert!(rank_pattern.is_empty());
    assert_eq!(init_lora_weights, LoraInitMode::ZerosB);
    assert_eq!(backbone_dtype, ComputePrecision::F32);
    assert_eq!(weight_decay, 0.01);
    assert_eq!(max_grad_norm, 1.0);
    assert!(!cached);
    assert_eq!(hard_negatives, HardNegativeConfig::default());
    assert_eq!(matryoshka_dims, Vec::<usize>::new());
    assert_eq!(seed, jammi_wire::fine_tune::DEFAULT_FINE_TUNE_SEED);
    assert_eq!(keep_last_n_checkpoints, None::<u32>);
}

/// A non-default value for every field, so a field whose non-default value
/// this test forgets to set would otherwise pass vacuously against the
/// all-default assertions above. Every field is asserted individually, and
/// the whole struct is re-asserted against itself after a `Clone`, so a field
/// missing from `PartialEq`/`Clone`'s derive (impossible for a derived impl,
/// but this is the oracle that would catch a future hand-written one that
/// dropped a field) would fail here too.
#[test]
fn every_field_holds_a_non_default_value_and_survives_a_clone() {
    let cfg = FineTuneConfig {
        lora_rank: 32,
        lora_alpha: 8.0,
        lora_dropout: 0.1,
        learning_rate: 1e-3,
        epochs: 5,
        batch_size: 16,
        max_seq_length: 256,
        embedding_loss: Some(EmbeddingLoss::Triplet { margin: 0.3 }),
        classification_loss: Some(ClassificationLoss::CrossEntropy),
        regression_loss: Some(RegressionLoss::Crps),
        quantile_levels: vec![0.1, 0.5, 0.9],
        gradient_accumulation_steps: 4,
        validation_fraction: 0.2,
        early_stopping_patience: 5,
        warmup_steps: 50,
        lr_schedule: LrSchedule::LinearDecay,
        early_stopping_metric: EarlyStoppingMetric::TrainLoss,
        target_modules: vec!["query".to_string(), "value".to_string()],
        layers_to_transform: Some(vec![0, 1, 2]),
        use_rslora: true,
        rank_pattern: std::collections::HashMap::from([("query".to_string(), 16)]),
        init_lora_weights: LoraInitMode::Gaussian,
        backbone_dtype: ComputePrecision::BF16,
        weight_decay: 0.05,
        max_grad_norm: 2.0,
        cached: true,
        hard_negatives: HardNegativeConfig {
            mine: true,
            k: 4,
            exclude_hops: 2,
            refresh_every: 3,
        },
        matryoshka_dims: vec![64, 128],
        seed: 1234,
        keep_last_n_checkpoints: Some(3),
    };

    let FineTuneConfig {
        lora_rank,
        lora_alpha,
        lora_dropout,
        learning_rate,
        epochs,
        batch_size,
        max_seq_length,
        embedding_loss,
        classification_loss,
        regression_loss,
        quantile_levels,
        gradient_accumulation_steps,
        validation_fraction,
        early_stopping_patience,
        warmup_steps,
        lr_schedule,
        early_stopping_metric,
        target_modules,
        layers_to_transform,
        use_rslora,
        rank_pattern,
        init_lora_weights,
        backbone_dtype,
        weight_decay,
        max_grad_norm,
        cached,
        hard_negatives,
        matryoshka_dims,
        seed,
        keep_last_n_checkpoints,
    } = cfg.clone();

    assert_eq!(lora_rank, 32);
    assert_eq!(lora_alpha, 8.0);
    assert_eq!(lora_dropout, 0.1);
    assert_eq!(learning_rate, 1e-3);
    assert_eq!(epochs, 5);
    assert_eq!(batch_size, 16);
    assert_eq!(max_seq_length, 256);
    assert_eq!(embedding_loss, Some(EmbeddingLoss::Triplet { margin: 0.3 }));
    assert_eq!(classification_loss, Some(ClassificationLoss::CrossEntropy));
    assert_eq!(regression_loss, Some(RegressionLoss::Crps));
    assert_eq!(quantile_levels, vec![0.1, 0.5, 0.9]);
    assert_eq!(gradient_accumulation_steps, 4);
    assert_eq!(validation_fraction, 0.2);
    assert_eq!(early_stopping_patience, 5);
    assert_eq!(warmup_steps, 50);
    assert_eq!(lr_schedule, LrSchedule::LinearDecay);
    assert_eq!(early_stopping_metric, EarlyStoppingMetric::TrainLoss);
    assert_eq!(
        target_modules,
        vec!["query".to_string(), "value".to_string()]
    );
    assert_eq!(layers_to_transform, Some(vec![0, 1, 2]));
    assert!(use_rslora);
    assert_eq!(
        rank_pattern,
        std::collections::HashMap::from([("query".to_string(), 16)])
    );
    assert_eq!(init_lora_weights, LoraInitMode::Gaussian);
    assert_eq!(backbone_dtype, ComputePrecision::BF16);
    assert_eq!(weight_decay, 0.05);
    assert_eq!(max_grad_norm, 2.0);
    assert!(cached);
    assert_eq!(
        hard_negatives,
        HardNegativeConfig {
            mine: true,
            k: 4,
            exclude_hops: 2,
            refresh_every: 3,
        }
    );
    assert_eq!(matryoshka_dims, vec![64, 128]);
    assert_eq!(seed, 1234);
    assert_eq!(keep_last_n_checkpoints, Some(3));

    assert_eq!(
        FineTuneConfig {
            lora_rank,
            lora_alpha,
            lora_dropout,
            learning_rate,
            epochs,
            batch_size,
            max_seq_length,
            embedding_loss,
            classification_loss,
            regression_loss,
            quantile_levels,
            gradient_accumulation_steps,
            validation_fraction,
            early_stopping_patience,
            warmup_steps,
            lr_schedule,
            early_stopping_metric,
            target_modules,
            layers_to_transform,
            use_rslora,
            rank_pattern,
            init_lora_weights,
            backbone_dtype,
            weight_decay,
            max_grad_norm,
            cached,
            hard_negatives,
            matryoshka_dims,
            seed,
            keep_last_n_checkpoints,
        },
        cfg
    );
}
