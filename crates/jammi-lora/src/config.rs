//! Per-call LoRA build configuration and module-name matching helpers.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::init::LoraInitMode;

/// Borrowed-reference build configuration passed to encoder builders.
///
/// Designed to be constructed on the stack at each call site from a caller-
/// owned config object, avoiding clones of the underlying `Vec`s and `HashMap`s.
#[derive(Debug, Clone, Copy)]
pub struct LoraBuildConfig<'a> {
    /// Module-name suffixes that receive a LoRA adapter. `"all-linear"` is a
    /// wildcard that matches every linear layer.
    pub target_modules: &'a [String],
    /// Optional restriction to specific layer indices.
    pub layers_to_transform: &'a Option<Vec<usize>>,
    /// Default LoRA rank (per-module overrides come from `rank_pattern`).
    pub lora_rank: usize,
    /// LoRA scaling factor.
    pub lora_alpha: f64,
    /// Use RSLoRA-style `alpha / sqrt(rank)` scaling instead of `alpha / rank`.
    pub use_rslora: bool,
    /// Dropout probability applied to the LoRA path while training.
    pub lora_dropout: Option<f32>,
    /// Per-module rank overrides keyed by module-name substring.
    pub rank_pattern: &'a HashMap<String, usize>,
    /// How the LoRA A/B matrices are initialised.
    pub init_mode: LoraInitMode,
    /// Run seed for the LoRA A/B init and the dropout mask. Every adapter draw
    /// is a pure function of this seed and the parameter's fully-qualified name,
    /// so the same seed reproduces byte-identical adapters across processes.
    pub seed: u64,
}

static EMPTY_TARGETS: &[String] = &[];
static EMPTY_LAYERS: Option<Vec<usize>> = None;
static EMPTY_RANK_PATTERN: LazyLock<HashMap<String, usize>> = LazyLock::new(HashMap::new);

impl LoraBuildConfig<'static> {
    /// "No LoRA" convenience: empty `target_modules`, no dropout, `ZerosB` init.
    /// Used by inference paths and parity tests that load an encoder without
    /// any adapter installed.
    pub fn frozen() -> Self {
        Self {
            target_modules: EMPTY_TARGETS,
            layers_to_transform: &EMPTY_LAYERS,
            lora_rank: 0,
            lora_alpha: 0.0,
            use_rslora: false,
            lora_dropout: None,
            rank_pattern: &EMPTY_RANK_PATTERN,
            init_mode: LoraInitMode::ZerosB,
            // A frozen encoder installs no adapter, so the seed is never drawn
            // from; a fixed value keeps `frozen()` const-constructible.
            seed: 0,
        }
    }
}

/// Decide whether a linear layer at `layer_idx` with name suffix `module_name`
/// should receive a LoRA adapter.
///
/// Semantics (mirroring HuggingFace PEFT):
/// - If `layers_to_transform` is `Some(ids)`, `layer_idx` must be `Some(i)`
///   with `i` in `ids`.
/// - If `target_modules` contains `"all-linear"`, match every layer name.
/// - Otherwise `module_name` must equal or end with one of the target strings.
///
/// # `layer_idx: None` — an UNINDEXED site
///
/// Not every LoRA-wrappable linear sits inside a numbered repeating unit. A
/// tower's head-side projections (e.g. a CLAP audio projection's
/// `linear1`/`linear2`) live outside the block stack and have no layer index
/// at all. PEFT models exactly this case and refuses such a site whenever a
/// layer filter is active: `check_target_module_exists`
/// (`peft/src/peft/tuners/tuners_utils.py:2353-2389`) extracts the index with
/// `re.match(r".*?\.[^.]*\.(?P<idx>\d+)\.", key)` and, when the key has no
/// numbered segment, sets `layer_index = None` and then
/// `target_module_found = False` — i.e. "restrict to layers `[0, 2]`" cannot
/// be satisfied by a module that belongs to no layer.
///
/// This function follows that rule exactly:
/// - `(None, Some(_))` → `false`: a layer filter is active and this site has
///   no index to test against it. Answering `true` would silently widen a
///   caller's explicit "only these layers" restriction to also cover every
///   unindexed head site.
/// - `(None, None)` → fall through to the ordinary selector match: with no
///   filter active there is nothing for the missing index to fail.
pub fn should_apply_lora(
    module_name: &str,
    target_modules: &[String],
    layer_idx: Option<usize>,
    layers_to_transform: &Option<Vec<usize>>,
) -> bool {
    if let Some(indices) = layers_to_transform {
        // `None` (an unindexed site) can never satisfy an active layer
        // filter — see this function's own doc for the PEFT citation.
        match layer_idx {
            Some(idx) if indices.contains(&idx) => {}
            _ => return false,
        }
    }

    if target_modules.iter().any(|t| t == "all-linear") {
        return true;
    }
    target_modules
        .iter()
        .any(|t| module_name == t.as_str() || module_name.ends_with(t.as_str()))
}

/// Effective LoRA rank for `module_name`: consult `rank_pattern` for the first
/// substring match, falling back to `default_rank`.
pub fn effective_rank(
    module_name: &str,
    default_rank: usize,
    rank_pattern: &HashMap<String, usize>,
) -> usize {
    rank_pattern
        .iter()
        .find(|(k, _)| module_name.contains(k.as_str()))
        .map(|(_, &r)| r)
        .unwrap_or(default_rank)
}
