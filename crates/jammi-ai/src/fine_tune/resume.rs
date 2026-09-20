//! The durable resume bundle for a crashed-and-resumed LoRA fine-tune.
//!
//! A fine-tune that dies mid-training must continue the *exact* trajectory it
//! would have without the crash — not an approximation. On `Device::Cpu` the
//! forward+backward+step is a pure function of `(seed, source rows, config)`
//! (seeded init/dropout, ordered source read), so the only state a resume must
//! carry across the process boundary is the optimiser's full trajectory plus the
//! adapter weights and the run counters. This module is that bundle's
//! serialisation contract.
//!
//! ## What the bundle holds, and why each piece
//!
//! - **Adapter weights** (`adapter.safetensors`) — the LoRA A/B tensors. Without
//!   them a resume restarts from the seeded init, not the epoch-k weights.
//! - **Optimiser moments** (`optimizer.safetensors`) — AdamW's first/second
//!   moment per parameter **keyed by name**, plus the global step `t`. This is
//!   the piece weights-only checkpointing silently drops: zero moments and `t = 1`
//!   bias-correction make the first post-resume step diverge immediately, even
//!   when the weights match. The moments are name-keyed (never positional)
//!   because `AdamW`'s state order is `VarMap::all_vars()`'s HashMap order, which
//!   is not stable across processes — serialising positionally would silently load
//!   the wrong parameter's moments.
//! - **`resume_state.json`** — `(epoch, global_step, step_t, seed)`, the
//!   `TargetScaler`'s `(μ, σ)` (persisted, *never* recomputed on resume — a
//!   recompute over re-read rows would diverge if the source changed by a hair),
//!   and each dropout stream's draw position (so a resumed run replays the same
//!   masks the uninterrupted run drew).

use std::collections::HashMap;
use std::path::Path;

use bytes::Bytes;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use jammi_db::error::{JammiError, Result};
use jammi_lora::{ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};

/// The optimiser-moments safetensors file inside a resume bundle. Each parameter
/// `{name}` contributes `{name}.m` (first moment) and `{name}.v` (second moment).
const MOMENTS_FILE: &str = "optimizer.safetensors";
/// The run-state JSON inside a resume bundle.
const STATE_FILE: &str = "resume_state.json";

/// The current [`ResumeState`] schema version. Bumped whenever a field's
/// UNIT or MEANING changes in a way that would silently mis-restore a
/// checkpoint of another version if read as this schema. Version 2 stores
/// `dropout_positions` per RANK as per-FORWARD Philox counters (see that
/// field's doc). A bundle of any other version — including one with NO
/// `schema_version` field — is refused typed by
/// [`ResumeState::check_schema_version`]: this crate ships no reader for
/// any other shape, and a resume that silently restarted from scratch
/// would discard the trajectory the job's checkpoints exist to preserve.
pub const RESUME_STATE_SCHEMA_VERSION: u32 = 2;

/// The version an ABSENT `schema_version` key parses as (`#[serde(default)]`
/// on [`ResumeState::schema_version`]) — a checkpoint with no version field.
/// Never equal to a real [`RESUME_STATE_SCHEMA_VERSION`] (versions start at
/// 1), so the ONE version check refuses it with no separate "field missing"
/// branch; the refusal names it as unversioned.
const UNVERSIONED_SCHEMA_VERSION: u32 = 0;

/// `serde`'s `default` hook for [`ResumeState::schema_version`] — a function
/// because `#[serde(default = ...)]` names a path, not a literal.
fn unversioned_schema_version() -> u32 {
    UNVERSIONED_SCHEMA_VERSION
}

/// The non-tensor run state persisted alongside the weights and moments. Every
/// field is authoritative on resume — in particular `scaler` is *loaded*, never
/// recomputed, so a source mutated between crash and resume cannot perturb the
/// de-standardisation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResumeState {
    /// The schema version this bundle was written under — see
    /// [`RESUME_STATE_SCHEMA_VERSION`]. `#[serde(default)]` to
    /// `UNVERSIONED_SCHEMA_VERSION` (`0`): a checkpoint with no
    /// `schema_version` key in its JSON parses as version `0` rather than
    /// failing the whole deserialize, so [`Self::check_schema_version`]
    /// refuses it typed, naming it unversioned.
    #[serde(default = "unversioned_schema_version")]
    pub schema_version: u32,
    /// The last epoch whose optimizer steps all completed — the boundary this
    /// checkpoint was taken at. The resumed run starts at `epoch + 1`.
    pub last_completed_epoch: usize,
    /// The optimizer-step counter at the boundary (== `step_t` for a run with no
    /// divergence skips; tracked separately because the trainer's `global_step`
    /// is its own loop counter).
    pub global_step: usize,
    /// AdamW's internal step counter `t` — the bias-correction exponent the first
    /// post-resume step depends on.
    pub step_t: usize,
    /// The run seed, carried so a resumed run's seeded init/dropout derive from
    /// the same base as the original.
    pub seed: u64,
    /// The `TargetScaler`'s `(μ, σ)` for a regression run, or `None`. Persisted so
    /// resume loads the authoritative standardiser rather than recomputing it.
    pub scaler: Option<(f64, f64)>,
    /// Each RANK's own per-layer dropout FORWARD COUNTER at the boundary,
    /// keyed `rank -> {layer}.dropout -> counter`: each rank's own dropout
    /// position is gathered to rank 0 at the epoch boundary and stored PER RANK here, so a resumed
    /// gang at equal topology restores EVERY rank's own stream (rank `r`'s
    /// `TrainingLoop` sets its layers' counters from `dropout_positions[r]`,
    /// never rank 0's) and reproduces an uninterrupted run byte-for-byte. At
    /// `W = 1` this map always has exactly the one entry for rank `0`.
    ///
    /// A resumed run SETS each layer's counter to its own rank's persisted
    /// value (O(1), an assignment) so its next training
    /// forwards draw the same masks the uninterrupted run drew.
    ///
    /// **Unit:** per-FORWARD Philox counter values
    /// (`jammi_kernels::ops::DropoutFused`'s `forward_idx`) — one increment
    /// per training forward through the layer, regardless of the
    /// activation's element count; never a per-ELEMENT draw count. Reading
    /// a counter of one unit as the other is an off-by-many-orders-of-
    /// magnitude misinterpretation, which is why
    /// [`RESUME_STATE_SCHEMA_VERSION`] exists and a bundle of another
    /// version is refused typed.
    pub dropout_positions: HashMap<u32, HashMap<String, u64>>,
}

impl ResumeState {
    /// Refuse a bundle this binary has no reader for: `Ok(())` iff
    /// `schema_version` is [`RESUME_STATE_SCHEMA_VERSION`], otherwise
    /// [`JammiError::IncompatibleFormat`] naming the bundle's state file,
    /// the version found (or that none was stamped), and the one supported.
    /// A pure function over the bundle's header — the one version check.
    pub fn check_schema_version(&self) -> Result<()> {
        if self.schema_version == RESUME_STATE_SCHEMA_VERSION {
            return Ok(());
        }
        let found = match self.schema_version {
            UNVERSIONED_SCHEMA_VERSION => "no schema_version".to_string(),
            version => format!("schema_version {version}"),
        };
        Err(JammiError::IncompatibleFormat {
            artifact: STATE_FILE.to_string(),
            found,
            supported: format!("schema_version {RESUME_STATE_SCHEMA_VERSION}"),
        })
    }
}

/// AdamW first/second moment buffers per parameter, keyed by parameter name —
/// the order-independent correlation that lets the resume bundle serialize and
/// restore optimizer moments by name rather than by the unstable `all_vars()`
/// position.
pub type NamedMoments = HashMap<String, (Tensor, Tensor)>;

/// A restored resume bundle: the tensors and run state a resumed [`TrainingLoop`]
/// loads back into its target, optimizer, and counters.
///
/// [`TrainingLoop`]: crate::fine_tune::trainer::TrainingLoop
pub struct RestoredCheckpoint {
    /// LoRA A/B tensors keyed as `named_trainable_weights` produces them.
    pub weights: HashMap<String, Tensor>,
    /// Per-parameter `(first_moment, second_moment)` keyed by parameter name.
    pub moments: NamedMoments,
    /// The persisted run state.
    pub state: ResumeState,
}

/// Serialise one epoch's checkpoint to `(name, bytes)` pairs ready for
/// `ArtifactStore::stage_checkpoint`: the loadable adapter — `weights`, the
/// A/B tensors, plus `adapter_config`, the adapter's own metadata — written
/// through `jammi_lora::save_adapter` exactly as the served final adapter
/// is, so the bundle loads for inference by the same path; then the AdamW
/// `moments`, keyed by the *same* parameter names (the trainer correlates
/// positions to names from the single `all_vars()` snapshot before calling
/// this), and the run `state` a resume restores. The safetensors files are
/// serialised through a scratch dir (candle serialises tensors only to a
/// path), then read back as bytes.
pub fn capture_bundle<C: Serialize>(
    scratch_dir: &Path,
    weights: &HashMap<String, Tensor>,
    adapter_config: &C,
    moments: &NamedMoments,
    state: &ResumeState,
) -> Result<Vec<(String, Bytes)>> {
    std::fs::create_dir_all(scratch_dir)?;

    jammi_lora::save_adapter(scratch_dir, weights, adapter_config)
        .map_err(|e| JammiError::FineTune(format!("checkpoint: save adapter: {e}")))?;
    let weights_path = scratch_dir.join(ADAPTER_WEIGHTS_FILE);
    let adapter_config_path = scratch_dir.join(ADAPTER_CONFIG_FILE);

    // Flatten the per-parameter moment pair into a single name-keyed map:
    // `{name}.m` / `{name}.v`. The `.m`/`.v` suffix cannot collide with a real
    // parameter name because every adapter tensor ends in `.lora_a` / `.lora_b`.
    let mut moment_tensors: HashMap<String, Tensor> = HashMap::with_capacity(moments.len() * 2);
    for (name, (m, v)) in moments {
        moment_tensors.insert(format!("{name}.m"), m.clone());
        moment_tensors.insert(format!("{name}.v"), v.clone());
    }
    let moments_path = scratch_dir.join(MOMENTS_FILE);
    candle_core::safetensors::save(&moment_tensors, &moments_path)
        .map_err(|e| JammiError::FineTune(format!("resume: save moments: {e}")))?;

    let state_bytes = serde_json::to_vec(state)
        .map_err(|e| JammiError::FineTune(format!("resume: serialize state: {e}")))?;

    Ok(vec![
        (
            ADAPTER_WEIGHTS_FILE.to_string(),
            Bytes::from(std::fs::read(&weights_path)?),
        ),
        (
            ADAPTER_CONFIG_FILE.to_string(),
            Bytes::from(std::fs::read(&adapter_config_path)?),
        ),
        (
            MOMENTS_FILE.to_string(),
            Bytes::from(std::fs::read(&moments_path)?),
        ),
        (STATE_FILE.to_string(), Bytes::from(state_bytes)),
    ])
}

/// Load a resume bundle from a fetched [`jammi_db::store::LocalArtifact`]
/// directory, reconstructing the weights, name-keyed moments, and run
/// state. A bundle of another schema version is the typed refusal
/// [`ResumeState::check_schema_version`] makes, before the (larger)
/// safetensors files are touched: nothing else in such a bundle is usable.
/// A moments file with a `{name}.m` lacking its `{name}.v` (or vice versa)
/// is refused too — a torn optimizer state must not restore half a
/// parameter's trajectory and silently zero the rest.
pub fn load_bundle(dir: &Path, device: &Device) -> Result<RestoredCheckpoint> {
    let state_bytes = std::fs::read(dir.join(STATE_FILE))?;
    let state: ResumeState = serde_json::from_slice(&state_bytes)
        .map_err(|e| JammiError::FineTune(format!("resume: parse state: {e}")))?;
    state.check_schema_version()?;

    let weights = candle_core::safetensors::load(dir.join(ADAPTER_WEIGHTS_FILE), device)
        .map_err(|e| JammiError::FineTune(format!("resume: load weights: {e}")))?;

    let moment_tensors = candle_core::safetensors::load(dir.join(MOMENTS_FILE), device)
        .map_err(|e| JammiError::FineTune(format!("resume: load moments: {e}")))?;
    let moments = pair_moments(moment_tensors)?;

    Ok(RestoredCheckpoint {
        weights,
        moments,
        state,
    })
}

/// Reassemble the flat `{name}.m` / `{name}.v` map into per-parameter pairs,
/// erroring on a half-present parameter.
fn pair_moments(flat: HashMap<String, Tensor>) -> Result<NamedMoments> {
    let mut first: HashMap<String, Tensor> = HashMap::new();
    let mut second: HashMap<String, Tensor> = HashMap::new();
    for (key, tensor) in flat {
        if let Some(name) = key.strip_suffix(".m") {
            first.insert(name.to_string(), tensor);
        } else if let Some(name) = key.strip_suffix(".v") {
            second.insert(name.to_string(), tensor);
        } else {
            return Err(JammiError::FineTune(format!(
                "resume: optimizer moment key '{key}' is neither a '.m' nor a '.v'"
            )));
        }
    }
    if first.len() != second.len() {
        return Err(JammiError::FineTune(format!(
            "resume: {} first-moments but {} second-moments — torn optimizer state",
            first.len(),
            second.len()
        )));
    }
    let mut out = HashMap::with_capacity(first.len());
    for (name, m) in first {
        let v = second.remove(&name).ok_or_else(|| {
            JammiError::FineTune(format!(
                "resume: parameter '{name}' has a first moment but no second moment"
            ))
        })?;
        out.insert(name, (m, v));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(device: &Device, v: f32) -> Tensor {
        Tensor::from_vec(vec![v, v + 1.0, v + 2.0], (3,), device).unwrap()
    }

    #[test]
    fn bundle_round_trips_weights_moments_and_state() {
        let device = Device::Cpu;
        let scratch = tempfile::tempdir().unwrap();

        let mut weights = HashMap::new();
        weights.insert("projection.lora_a".to_string(), tiny(&device, 1.0));
        weights.insert("projection.lora_b".to_string(), tiny(&device, 4.0));

        let mut moments = HashMap::new();
        moments.insert(
            "projection.lora_a".to_string(),
            (tiny(&device, 10.0), tiny(&device, 20.0)),
        );
        moments.insert(
            "projection.lora_b".to_string(),
            (tiny(&device, 30.0), tiny(&device, 40.0)),
        );

        let mut rank0_positions = HashMap::new();
        rank0_positions.insert("projection.dropout".to_string(), 96);
        let mut rank1_positions = HashMap::new();
        rank1_positions.insert("projection.dropout".to_string(), 57);
        let mut dropout_positions = HashMap::new();
        dropout_positions.insert(0u32, rank0_positions);
        dropout_positions.insert(1u32, rank1_positions);
        let state = ResumeState {
            schema_version: RESUME_STATE_SCHEMA_VERSION,
            last_completed_epoch: 2,
            global_step: 7,
            step_t: 7,
            seed: 42,
            scaler: Some((2017.0, 2.5)),
            dropout_positions,
        };

        let bundle = capture_bundle(
            scratch.path(),
            &weights,
            &fixture_config(),
            &moments,
            &state,
        )
        .unwrap();
        // Materialise the bundle to a dir as the artifact store would, then reload.
        let out = tempfile::tempdir().unwrap();
        for (name, bytes) in &bundle {
            std::fs::write(out.path().join(name), bytes).unwrap();
        }
        let restored = load_bundle(out.path(), &device).unwrap();

        assert_eq!(restored.state, state);
        for (name, t) in &weights {
            let got: Vec<f32> = restored.weights[name].to_vec1().unwrap();
            let want: Vec<f32> = t.to_vec1().unwrap();
            assert_eq!(got, want, "weight '{name}' did not round-trip");
        }
        for (name, (m, v)) in &moments {
            let (rm, rv) = &restored.moments[name];
            assert_eq!(
                rm.to_vec1::<f32>().unwrap(),
                m.to_vec1::<f32>().unwrap(),
                "first moment '{name}' did not round-trip"
            );
            assert_eq!(
                rv.to_vec1::<f32>().unwrap(),
                v.to_vec1::<f32>().unwrap(),
                "second moment '{name}' did not round-trip"
            );
        }
    }

    #[test]
    fn torn_moments_are_a_hard_error() {
        let device = Device::Cpu;
        // A first moment with no matching second moment.
        let mut flat = HashMap::new();
        flat.insert("p.m".to_string(), tiny(&device, 1.0));
        let err = pair_moments(flat).unwrap_err().to_string();
        assert!(err.contains("torn optimizer state") || err.contains("no second moment"));
    }

    /// A minimal, valid (weights + moments) bundle, real safetensors files
    /// but written through `capture_bundle` so both non-state files are
    /// trivially valid — only `resume_state.json` gets doctored by the two
    /// schema-version tests below.
    fn minimal_versioned_bundle(scratch: &std::path::Path) -> Vec<(String, Bytes)> {
        let device = Device::Cpu;
        let mut weights = HashMap::new();
        weights.insert("w.lora_a".to_string(), tiny(&device, 1.0));
        weights.insert("w.lora_b".to_string(), tiny(&device, 2.0));
        let mut moments = HashMap::new();
        moments.insert(
            "w.lora_a".to_string(),
            (tiny(&device, 3.0), tiny(&device, 4.0)),
        );
        moments.insert(
            "w.lora_b".to_string(),
            (tiny(&device, 5.0), tiny(&device, 6.0)),
        );
        let state = ResumeState {
            schema_version: RESUME_STATE_SCHEMA_VERSION,
            last_completed_epoch: 0,
            global_step: 0,
            step_t: 0,
            seed: 1,
            scaler: None,
            dropout_positions: HashMap::new(),
        };
        capture_bundle(scratch, &weights, &fixture_config(), &moments, &state).unwrap()
    }

    /// The adapter metadata a checkpoint carries beside its weights; opaque
    /// to everything a resume reads.
    fn fixture_config() -> serde_json::Value {
        serde_json::json!({ "lora_rank": 2, "head_layers": ["projection"] })
    }

    /// The typed refusal `load_bundle` makes of `bundle` written to a dir,
    /// with `resume_state.json` rewritten by `doctor`.
    fn refusal_of(
        bundle: &[(String, Bytes)],
        doctor: impl Fn(&mut serde_json::Map<String, serde_json::Value>),
    ) -> JammiError {
        let out = tempfile::tempdir().unwrap();
        for (name, bytes) in bundle {
            if name == STATE_FILE {
                let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                doctor(
                    value
                        .as_object_mut()
                        .expect("resume_state.json must serialize as a JSON object"),
                );
                std::fs::write(out.path().join(name), serde_json::to_vec(&value).unwrap()).unwrap();
            } else {
                std::fs::write(out.path().join(name), bytes).unwrap();
            }
        }
        load_bundle(out.path(), &Device::Cpu)
            .err()
            .expect("a bundle of another schema version is refused")
    }

    /// A checkpoint whose `schema_version` is PRESENT but not this
    /// binary's is refused typed — `IncompatibleFormat` naming the state
    /// file, the version found and the one supported — never silently
    /// restored under an incompatible `dropout_positions` shape and never a
    /// from-scratch restart.
    #[test]
    fn a_bundle_of_another_schema_version_is_a_typed_refusal() {
        let scratch = tempfile::tempdir().unwrap();
        let bundle = minimal_versioned_bundle(scratch.path());
        let foreign = RESUME_STATE_SCHEMA_VERSION + 1;
        match refusal_of(&bundle, |state| {
            state.insert("schema_version".into(), serde_json::Value::from(foreign));
        }) {
            JammiError::IncompatibleFormat {
                artifact,
                found,
                supported,
            } => {
                assert_eq!(artifact, STATE_FILE);
                assert_eq!(found, format!("schema_version {foreign}"));
                assert_eq!(
                    supported,
                    format!("schema_version {RESUME_STATE_SCHEMA_VERSION}")
                );
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    /// A checkpoint with NO `schema_version` field at all (not merely
    /// `0`/`null`) is the same typed refusal, naming it unversioned.
    #[test]
    fn an_unversioned_bundle_is_a_typed_refusal() {
        let scratch = tempfile::tempdir().unwrap();
        let bundle = minimal_versioned_bundle(scratch.path());
        match refusal_of(&bundle, |state| {
            state.remove("schema_version");
        }) {
            JammiError::IncompatibleFormat { found, .. } => {
                assert_eq!(found, "no schema_version");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    /// The check is a pure function over the header: the table of versions
    /// around the supported one.
    #[test]
    fn check_schema_version_admits_only_this_binarys_version() {
        let state = |schema_version| ResumeState {
            schema_version,
            last_completed_epoch: 0,
            global_step: 0,
            step_t: 0,
            seed: 1,
            scaler: None,
            dropout_positions: HashMap::new(),
        };
        assert!(state(RESUME_STATE_SCHEMA_VERSION)
            .check_schema_version()
            .is_ok());
        for foreign in [
            0,
            RESUME_STATE_SCHEMA_VERSION - 1,
            RESUME_STATE_SCHEMA_VERSION + 1,
        ] {
            assert!(
                matches!(
                    state(foreign).check_schema_version(),
                    Err(JammiError::IncompatibleFormat { .. })
                ),
                "schema_version {foreign}"
            );
        }
    }

    /// A genuinely torn moments file (unrelated to schema versioning) is
    /// its own refusal once the version check passes.
    #[test]
    fn a_same_version_bundle_with_torn_moments_is_still_a_hard_error() {
        let device = Device::Cpu;
        let scratch = tempfile::tempdir().unwrap();
        let bundle = minimal_versioned_bundle(scratch.path());
        let out = tempfile::tempdir().unwrap();
        for (name, bytes) in &bundle {
            if name == MOMENTS_FILE {
                // Overwrite with a moments file missing every `.v` entry —
                // torn, but still a validly-versioned bundle otherwise.
                let mut only_m = HashMap::new();
                only_m.insert("w.lora_a.m".to_string(), tiny(&device, 1.0));
                only_m.insert("w.lora_b.m".to_string(), tiny(&device, 2.0));
                let torn_path = out.path().join(MOMENTS_FILE);
                candle_core::safetensors::save(&only_m, &torn_path).unwrap();
            } else {
                std::fs::write(out.path().join(name), bytes).unwrap();
            }
        }
        let err = load_bundle(out.path(), &device)
            .err()
            .expect("a torn moments file is refused");
        let msg = err.to_string();
        assert!(
            msg.contains("torn optimizer state") || msg.contains("no second moment"),
            "error must name the torn-moments failure: {msg}"
        );
    }
}
