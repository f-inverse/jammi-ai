//! `AnyEncoder::lora_site_names` is the LoRA selector vocabulary each
//! encoder family accepts, and this module is what makes it a MEASURED
//! claim rather than a hand-maintained list.
//!
//! # Why the API exists
//!
//! `jammi_lora::should_apply_lora` has no notion of a wrong name. A
//! `target_modules` list of plausible-but-wrong strings (`q_proj` against a
//! BERT checkpoint, `query` against ModernBERT's fused `Wqkv`) matches
//! nothing, every site stays `MaybeLoraLinear::Frozen`, and the build
//! SUCCEEDS with zero trainable parameters — a run that then trains happily
//! and updates nothing. Naming the real sites is what lets a caller refuse
//! that build up front.
//!
//! # What is asserted, per variant
//!
//! A hardcoded list of strings in a doc comment is exactly the thing that
//! goes stale, so nothing here trusts the constant:
//!
//! 1. every name in `lora_site_names()`, used ALONE in `target_modules`,
//!    selects at least one real site on a real fixture — measured as the
//!    adapter keys `named_trainable_weights()` reports, not as a count
//!    somebody predicted;
//! 2. every such selection is a SUBSET of what `all-linear` selects (a name
//!    cannot reach a site the wildcard misses), and the UNION over all names
//!    is exactly the `all-linear` set (the vocabulary is complete — no site
//!    is reachable only by the wildcard);
//! 3. the vocabulary has no duplicate entries and is non-empty;
//! 4. NEGATIVE CONTROL: a name that is in no vocabulary selects nothing, so
//!    "the keys are non-empty" is a real discriminator and not something the
//!    measurement returns unconditionally.
//!
//! Point 2's union half is the one that catches the interesting regression:
//! adding a LoRA site to a family and forgetting to name it leaves every
//! per-name build green while `all-linear` quietly grows past their union.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarMap;
use jammi_encoders::{
    AnyEncoder, Bert, BertConfig, ClipText, ClipTextConfig, DistilBert, HtsatAudio,
    HtsatAudioConfig, ModernBert, ModernBertConfig, OpenClipVisionConfig,
    OpenClipVisionTransformer, Pooling,
};
use jammi_lora::{LoraBuildConfig, LoraInitMode};

/// A name no family uses and no real site name ends with — so
/// `should_apply_lora`'s `ends_with` arm cannot match it either.
const NOT_A_SITE: &str = "definitely_not_a_lora_site";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Owns the `target_modules` strings a [`LoraBuildConfig`] borrows.
struct Targets {
    targets: Vec<String>,
    layers: Option<Vec<usize>>,
    rank_pattern: HashMap<String, usize>,
}

impl Targets {
    fn new(targets: &[&str]) -> Self {
        Self {
            targets: targets.iter().map(|s| (*s).to_string()).collect(),
            // No layer filter: HTSAT's two projection-head sites are
            // UNINDEXED, and an active `layers_to_transform` refuses those
            // by design (`should_apply_lora`'s own doc), which would make
            // this oracle report a false "selects nothing".
            layers: None,
            rank_pattern: HashMap::new(),
        }
    }

    fn config(&self) -> LoraBuildConfig<'_> {
        LoraBuildConfig {
            target_modules: &self.targets,
            layers_to_transform: &self.layers,
            lora_rank: 2,
            lora_alpha: 4.0,
            use_rslora: false,
            lora_dropout: None,
            rank_pattern: &self.rank_pattern,
            // `ZerosB` is fine here: nothing runs a backward. This oracle
            // is about WHICH sites exist, never about their values.
            init_mode: LoraInitMode::ZerosB,
            seed: 0x51_7e5,
        }
    }
}

/// The adapter keys a build with these `target_modules` actually installed.
fn selected_keys(encoder: &AnyEncoder) -> BTreeSet<String> {
    encoder
        .named_trainable_weights()
        .expect("named_trainable_weights")
        .into_keys()
        .collect()
}

/// The four claims of this module's doc, run against one variant's real
/// fixture. `build` must produce a fresh encoder for the supplied
/// `target_modules` every call.
fn assert_lora_site_name_vocabulary(what: &str, build: &dyn Fn(&[&str]) -> AnyEncoder) {
    let all_linear = build(&["all-linear"]);
    let names = all_linear.lora_site_names();

    assert!(
        !names.is_empty(),
        "{what}: every variant carries real LoRA sites, so its vocabulary must be non-empty"
    );
    let unique: BTreeSet<&str> = names.iter().copied().collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "{what}: lora_site_names must not repeat a selector name, got {names:?}"
    );

    let all_keys = selected_keys(&all_linear);
    assert!(
        !all_keys.is_empty(),
        "{what}: `all-linear` must select at least one site, or this oracle is vacuous"
    );

    // Negative control FIRST: if the measurement returned a non-empty set
    // for a nonsense name, every per-name assertion below would be
    // meaningless.
    assert!(
        selected_keys(&build(&[NOT_A_SITE])).is_empty(),
        "{what}: `{NOT_A_SITE}` matches no site, so it must install NO adapter — the \
         per-name assertions below are only meaningful if this can come back empty"
    );

    let mut union: BTreeSet<String> = BTreeSet::new();
    for name in names {
        let keys = selected_keys(&build(&[name]));
        assert!(
            !keys.is_empty(),
            "{what}: `{name}` is in lora_site_names() but selects NO site on this fixture"
        );
        assert!(
            keys.is_subset(&all_keys),
            "{what}: `{name}` selected sites `all-linear` does not: {:?}",
            keys.difference(&all_keys).collect::<Vec<_>>()
        );
        union.extend(keys);
    }

    assert_eq!(
        union, all_keys,
        "{what}: the union of every named selector must be exactly what `all-linear` selects \
         — a site reachable only by the wildcard is a site the vocabulary forgot to name"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-variant fixtures
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bert_lora_site_names_are_exactly_its_selectable_sites() {
    let device = Device::Cpu;
    let dir = repo_root().join("cookbook/fixtures/tiny_bert");
    let config: BertConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let weights = dir.join("model.safetensors");

    let build = |targets: &[&str]| -> AnyEncoder {
        let t = Targets::new(targets);
        let varmap = VarMap::new();
        AnyEncoder::Bert(
            Bert::builder()
                .pooling(Pooling::Mean)
                .lora(t.config())
                .backbone_dtype(DType::F32)
                .adapter(None)
                .build(&[weights.as_path()], &config, &device, &varmap)
                .expect("build Bert on tiny_bert"),
        )
    };
    assert_lora_site_name_vocabulary("bert", &build);
}

#[test]
fn distilbert_lora_site_names_are_exactly_its_selectable_sites() {
    let device = Device::Cpu;
    // The same synthetic checkpoint `crate::distilbert`'s own oracles use —
    // this crate ships no DistilBERT HF fixture (see that module's doc).
    let config = crate::distilbert::tiny_config();
    let (_dir, weights) = crate::distilbert::write_synthetic_weights(&config, &device);

    let build = |targets: &[&str]| -> AnyEncoder {
        let t = Targets::new(targets);
        let varmap = VarMap::new();
        AnyEncoder::DistilBert(
            DistilBert::builder()
                .pooling(Pooling::Mean)
                .lora(t.config())
                .backbone_dtype(DType::F32)
                .adapter(None)
                .build(&[weights.as_path()], &config, &device, &varmap)
                .expect("build DistilBert on the synthetic checkpoint"),
        )
    };
    assert_lora_site_name_vocabulary("distilbert", &build);
}

#[test]
fn modernbert_lora_site_names_are_exactly_its_selectable_sites() {
    let device = Device::Cpu;
    let dir = repo_root().join("cookbook/fixtures/tiny_modernbert_classifier");
    let config: ModernBertConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let weights = dir.join("model.safetensors");

    let build = |targets: &[&str]| -> AnyEncoder {
        let t = Targets::new(targets);
        let varmap = VarMap::new();
        AnyEncoder::ModernBert(
            ModernBert::builder()
                .pooling(Pooling::Mean)
                .lora(t.config())
                .backbone_dtype(DType::F32)
                .adapter(None)
                .build(&[weights.as_path()], &config, &device, &varmap)
                .expect("build ModernBert on tiny_modernbert_classifier"),
        )
    };
    assert_lora_site_name_vocabulary("modernbert", &build);
}

fn open_clip_json() -> serde_json::Value {
    let path = repo_root().join("tests/fixtures/tiny_open_clip/open_clip_config.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn clip_text_lora_site_names_are_exactly_its_selectable_sites() {
    let device = Device::Cpu;
    let config = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let weights = repo_root().join("tests/fixtures/tiny_open_clip/open_clip_model.safetensors");

    let build = |targets: &[&str]| -> AnyEncoder {
        let t = Targets::new(targets);
        let varmap = VarMap::new();
        AnyEncoder::ClipText(
            ClipText::builder()
                .lora(t.config())
                .backbone_dtype(DType::F32)
                .build(&[weights.as_path()], &config, &device, &varmap)
                .expect("build ClipText on tiny_open_clip"),
        )
    };
    assert_lora_site_name_vocabulary("clip_text", &build);
}

#[test]
fn open_clip_vision_lora_site_names_are_exactly_its_selectable_sites() {
    let device = Device::Cpu;
    let config = OpenClipVisionConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let weights = repo_root().join("tests/fixtures/tiny_open_clip/open_clip_model.safetensors");

    let build = |targets: &[&str]| -> AnyEncoder {
        let t = Targets::new(targets);
        let varmap = VarMap::new();
        AnyEncoder::OpenClipVision(
            OpenClipVisionTransformer::builder()
                .lora(t.config())
                .backbone_dtype(DType::F32)
                .build(&[weights.as_path()], &config, &device, &varmap)
                .expect("build OpenClipVisionTransformer on tiny_open_clip"),
        )
    };
    assert_lora_site_name_vocabulary("open_clip_vision", &build);
}

#[test]
fn htsat_lora_site_names_are_exactly_its_selectable_sites() {
    let device = Device::Cpu;
    let dir = repo_root().join("cookbook/fixtures/htsat_clap_tiny");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let config = HtsatAudioConfig::from_hf_clap_config(&json).unwrap();
    let weights = dir.join("model.safetensors");

    let build = |targets: &[&str]| -> AnyEncoder {
        let t = Targets::new(targets);
        let varmap = VarMap::new();
        AnyEncoder::Htsat(Box::new(
            HtsatAudio::builder()
                .lora(t.config())
                .backbone_dtype(DType::F32)
                .build(&[weights.as_path()], &config, &device, &varmap)
                .expect("build HtsatAudio on htsat_clap_tiny"),
        ))
    };
    assert_lora_site_name_vocabulary("htsat_audio", &build);
}

/// The two OpenCLIP towers load the SAME residual block, so they must report
/// the same vocabulary — and it must be the same `&'static` slice, not two
/// lists that happen to agree today. Guards against a future "fix" that
/// gives one tower its own copy and lets the two drift.
#[test]
fn both_open_clip_towers_report_one_shared_vocabulary() {
    let device = Device::Cpu;
    let json = open_clip_json();
    let tcfg = ClipTextConfig::from_open_clip_config(&json).unwrap();
    let vcfg = OpenClipVisionConfig::from_open_clip_config(&json).unwrap();
    let weights = repo_root().join("tests/fixtures/tiny_open_clip/open_clip_model.safetensors");

    let t = Targets::new(&[]);
    let tvm = VarMap::new();
    let text = AnyEncoder::ClipText(
        ClipText::builder()
            .lora(t.config())
            .backbone_dtype(DType::F32)
            .build(&[weights.as_path()], &tcfg, &device, &tvm)
            .unwrap(),
    );
    let vvm = VarMap::new();
    let vision = AnyEncoder::OpenClipVision(
        OpenClipVisionTransformer::builder()
            .lora(t.config())
            .backbone_dtype(DType::F32)
            .build(&[weights.as_path()], &vcfg, &device, &vvm)
            .unwrap(),
    );

    assert!(
        std::ptr::eq(text.lora_site_names(), vision.lora_site_names()),
        "the two OpenCLIP towers must return the one shared vocabulary, not two equal copies"
    );
}

/// The six vocabularies are not interchangeable: the families name their
/// sites in their own checkpoints' terms. Asserted so a future
/// `lora_site_names` that collapsed to one generic list (or to a wrong arm
/// of the match) fails loudly instead of handing a caller names that select
/// nothing on the encoder it actually holds.
#[test]
fn the_bert_family_vocabularies_are_pairwise_distinct() {
    let device = Device::Cpu;

    let bert_dir = repo_root().join("cookbook/fixtures/tiny_bert");
    let bert_config: BertConfig =
        serde_json::from_str(&std::fs::read_to_string(bert_dir.join("config.json")).unwrap())
            .unwrap();
    let t = Targets::new(&[]);
    let bvm = VarMap::new();
    let bert = AnyEncoder::Bert(
        Bert::builder()
            .pooling(Pooling::Mean)
            .lora(t.config())
            .backbone_dtype(DType::F32)
            .adapter(None)
            .build(
                &[bert_dir.join("model.safetensors").as_path()],
                &bert_config,
                &device,
                &bvm,
            )
            .unwrap(),
    );

    let mb_dir = repo_root().join("cookbook/fixtures/tiny_modernbert_classifier");
    let mb_config: ModernBertConfig =
        serde_json::from_str(&std::fs::read_to_string(mb_dir.join("config.json")).unwrap())
            .unwrap();
    let mvm = VarMap::new();
    let modernbert = AnyEncoder::ModernBert(
        ModernBert::builder()
            .pooling(Pooling::Mean)
            .lora(t.config())
            .backbone_dtype(DType::F32)
            .adapter(None)
            .build(
                &[mb_dir.join("model.safetensors").as_path()],
                &mb_config,
                &device,
                &mvm,
            )
            .unwrap(),
    );

    let d_config = crate::distilbert::tiny_config();
    let (_dir, d_weights) = crate::distilbert::write_synthetic_weights(&d_config, &device);
    let dvm = VarMap::new();
    let distilbert = AnyEncoder::DistilBert(
        DistilBert::builder()
            .pooling(Pooling::Mean)
            .lora(t.config())
            .backbone_dtype(DType::F32)
            .adapter(None)
            .build(&[d_weights.as_path()], &d_config, &device, &dvm)
            .unwrap(),
    );

    let vocabularies = [
        ("bert", bert.lora_site_names()),
        ("distilbert", distilbert.lora_site_names()),
        ("modernbert", modernbert.lora_site_names()),
    ];
    for (i, (a_name, a)) in vocabularies.iter().enumerate() {
        for (b_name, b) in vocabularies.iter().skip(i + 1) {
            let a: BTreeSet<&str> = a.iter().copied().collect();
            let b: BTreeSet<&str> = b.iter().copied().collect();
            assert!(
                a.is_disjoint(&b),
                "{a_name} and {b_name} must not share a selector name, but both list {:?}",
                a.intersection(&b).collect::<Vec<_>>()
            );
        }
    }
}
