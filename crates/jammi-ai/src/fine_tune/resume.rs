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

/// The adapter-weights safetensors file inside a resume bundle.
const WEIGHTS_FILE: &str = "adapter.safetensors";
/// The optimiser-moments safetensors file inside a resume bundle. Each parameter
/// `{name}` contributes `{name}.m` (first moment) and `{name}.v` (second moment).
const MOMENTS_FILE: &str = "optimizer.safetensors";
/// The run-state JSON inside a resume bundle.
const STATE_FILE: &str = "resume_state.json";

/// The current [`ResumeState`] schema version. Bumped whenever a field's
/// UNIT or MEANING changes in a way that would silently mis-restore an
/// older checkpoint if read as the new schema — the C7 commit that
/// introduces this constant is itself such a bump: `dropout_positions`
/// changes unit from per-ELEMENT draw counts to per-FORWARD Philox
/// counters (closing esc-032/esc-033; see that field's own doc). See
/// [`load_bundle`]'s version check for how a mismatch (including a
/// checkpoint with NO `schema_version` field at all, from before this
/// commit) is refused rather than silently misinterpreted.
pub const RESUME_STATE_SCHEMA_VERSION: u32 = 1;

/// The non-tensor run state persisted alongside the weights and moments. Every
/// field is authoritative on resume — in particular `scaler` is *loaded*, never
/// recomputed, so a source mutated between crash and resume cannot perturb the
/// de-standardisation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResumeState {
    /// The schema version this bundle was written under — see
    /// [`RESUME_STATE_SCHEMA_VERSION`]. Deliberately carries NO
    /// `#[serde(default)]`: a checkpoint captured before this field
    /// existed has no `schema_version` key in its JSON at all, so
    /// deserializing it FAILS with a typed error naming the missing
    /// field, rather than silently defaulting to today's schema and
    /// misinterpreting `dropout_positions`'s unit. [`load_bundle`] adds an
    /// explicit post-parse check for a PRESENT-but-different version on
    /// top of that (the case a missing-field serde error alone would not
    /// catch: a checkpoint from a future/other schema version).
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
    /// Each layer's dropout FORWARD COUNTER at the boundary, keyed
    /// `{layer}.dropout`. A resumed run SETS each layer's counter to this
    /// value (O(1), an assignment — closing esc-033) so its next training
    /// forwards draw the same masks the uninterrupted run drew.
    ///
    /// **Unit change (schema version 1, this commit):** before device-side
    /// Philox dropout, this counted per-ELEMENT draws from an advancing
    /// host RNG stream (each draw advancing the stream's position by the
    /// activation's element count); it now counts per-FORWARD Philox counter
    /// values (`jammi_kernels::ops::
    /// DropoutFused`'s `forward_idx`) — one increment per training
    /// forward through the layer, regardless of the activation's element
    /// count. A checkpoint written under the OLD unit would silently
    /// restore a draw count into a forward counter if read under the new
    /// schema (an off-by-many-orders-of-magnitude misinterpretation, not
    /// a merely-stale one) — this is exactly why [`RESUME_STATE_SCHEMA_
    /// VERSION`] exists and why an unversioned checkpoint is refused
    /// rather than silently accepted (esc-032).
    pub dropout_positions: HashMap<String, u64>,
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

/// Serialise a resume bundle to `(name, bytes)` pairs ready for
/// `ArtifactStore::put_resume_checkpoint`.
///
/// `weights` are the adapter A/B tensors; `moments` are the AdamW moments keyed
/// by the *same* parameter names (the trainer correlates positions to names from
/// the single `all_vars()` snapshot before calling this). The two safetensors
/// files are serialised through a scratch dir (candle serialises tensors only to
/// a path), then read back as bytes — the same `candle_core::safetensors::save`
/// path the scratch checkpoints use.
pub fn capture_bundle(
    scratch_dir: &Path,
    weights: &HashMap<String, Tensor>,
    moments: &NamedMoments,
    state: &ResumeState,
) -> Result<Vec<(String, Bytes)>> {
    std::fs::create_dir_all(scratch_dir)?;

    let weights_path = scratch_dir.join(WEIGHTS_FILE);
    candle_core::safetensors::save(weights, &weights_path)
        .map_err(|e| JammiError::FineTune(format!("resume: save weights: {e}")))?;

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
            WEIGHTS_FILE.to_string(),
            Bytes::from(std::fs::read(&weights_path)?),
        ),
        (
            MOMENTS_FILE.to_string(),
            Bytes::from(std::fs::read(&moments_path)?),
        ),
        (STATE_FILE.to_string(), Bytes::from(state_bytes)),
    ])
}

/// Load a resume bundle from a fetched [`jammi_db::store::LocalArtifact`]
/// directory, reconstructing the weights, name-keyed moments, and run state.
///
/// A moments file with a `{name}.m` lacking its `{name}.v` (or vice versa) is a
/// hard error — a torn optimizer state must not restore half a parameter's
/// trajectory and silently zero the rest.
pub fn load_bundle(dir: &Path, device: &Device) -> Result<RestoredCheckpoint> {
    let weights = candle_core::safetensors::load(dir.join(WEIGHTS_FILE), device)
        .map_err(|e| JammiError::FineTune(format!("resume: load weights: {e}")))?;

    let moment_tensors = candle_core::safetensors::load(dir.join(MOMENTS_FILE), device)
        .map_err(|e| JammiError::FineTune(format!("resume: load moments: {e}")))?;
    let moments = pair_moments(moment_tensors)?;

    let state_bytes = std::fs::read(dir.join(STATE_FILE))?;
    // `ResumeState::schema_version` has no `#[serde(default)]`, so a
    // checkpoint written before this field existed (no `schema_version` key
    // in its JSON at all) fails RIGHT HERE with a typed error naming the
    // missing field — refused, not silently misinterpreted (esc-032).
    let state: ResumeState = serde_json::from_slice(&state_bytes).map_err(|e| {
        JammiError::FineTune(format!(
            "resume: parse state: {e} — a checkpoint written before the schema_version field \
             existed (pre-C7) has no such field and is refused rather than silently restored: \
             its 'dropout_positions' counted per-ELEMENT draws, not the per-forward Philox \
             counters this schema now expects, so misreading one as the other would silently \
             desynchronize every training forward's dropout mask from the run it resumes"
        ))
    })?;
    // A PRESENT but DIFFERENT version — the case the missing-field parse
    // error above cannot catch (e.g. a checkpoint from a future schema
    // version this binary predates). Refused with the same "typed error
    // naming the field" contract, not guessed at.
    if state.schema_version != RESUME_STATE_SCHEMA_VERSION {
        return Err(JammiError::FineTune(format!(
            "resume: resume_state.json has schema_version {}, this binary expects {} — \
             refusing rather than guessing how to interpret an unrecognized schema's \
             'dropout_positions' units",
            state.schema_version, RESUME_STATE_SCHEMA_VERSION
        )));
    }

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

        let mut dropout_positions = HashMap::new();
        dropout_positions.insert("projection.dropout".to_string(), 96);
        let state = ResumeState {
            schema_version: RESUME_STATE_SCHEMA_VERSION,
            last_completed_epoch: 2,
            global_step: 7,
            step_t: 7,
            seed: 42,
            scaler: Some((2017.0, 2.5)),
            dropout_positions,
        };

        let bundle = capture_bundle(scratch.path(), &weights, &moments, &state).unwrap();
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
        capture_bundle(scratch, &weights, &moments, &state).unwrap()
    }

    /// esc-032, the exact oracle the C7 contract asks for: an UNVERSIONED
    /// checkpoint fixture (a `resume_state.json` from before the
    /// `schema_version` field existed — the field is ABSENT, not merely
    /// `0`/`null`, matching what a real pre-C7 checkpoint literally wrote)
    /// must be REFUSED, never silently restored under today's
    /// (incompatible) `dropout_positions` unit.
    #[test]
    fn unversioned_checkpoint_is_refused_not_silently_restored() {
        let device = Device::Cpu;
        let scratch = tempfile::tempdir().unwrap();
        let bundle = minimal_versioned_bundle(scratch.path());
        let out = tempfile::tempdir().unwrap();
        for (name, bytes) in &bundle {
            if name == STATE_FILE {
                let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                value
                    .as_object_mut()
                    .expect("resume_state.json must serialize as a JSON object")
                    .remove("schema_version");
                std::fs::write(out.path().join(name), serde_json::to_vec(&value).unwrap()).unwrap();
            } else {
                std::fs::write(out.path().join(name), bytes).unwrap();
            }
        }
        // `.expect_err`/`.unwrap_err` require `T: Debug`, which
        // `RestoredCheckpoint` (holding raw `Tensor`s) does not implement —
        // match explicitly instead of adding a `Debug` derive purely for
        // this test's sake.
        let err = match load_bundle(out.path(), &device) {
            Err(e) => e,
            Ok(_) => panic!("an unversioned checkpoint must be refused, not silently restored"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("schema_version"),
            "the typed error must name the missing field: {msg}"
        );
    }

    /// The complementary case a missing-field parse error alone cannot
    /// catch: a checkpoint whose `schema_version` is PRESENT but does not
    /// match this binary's expectation. Also refused, not guessed at.
    #[test]
    fn mismatched_schema_version_is_refused() {
        let device = Device::Cpu;
        let scratch = tempfile::tempdir().unwrap();
        let bundle = minimal_versioned_bundle(scratch.path());
        let out = tempfile::tempdir().unwrap();
        for (name, bytes) in &bundle {
            if name == STATE_FILE {
                let mut value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                value["schema_version"] = serde_json::Value::from(RESUME_STATE_SCHEMA_VERSION + 1);
                std::fs::write(out.path().join(name), serde_json::to_vec(&value).unwrap()).unwrap();
            } else {
                std::fs::write(out.path().join(name), bytes).unwrap();
            }
        }
        let err = match load_bundle(out.path(), &device) {
            Err(e) => e,
            Ok(_) => panic!("a mismatched schema_version must be refused"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("schema_version"),
            "error must name the field: {msg}"
        );
        assert!(
            msg.contains(&(RESUME_STATE_SCHEMA_VERSION + 1).to_string())
                && msg.contains(&RESUME_STATE_SCHEMA_VERSION.to_string()),
            "error must state both the checkpoint's version and this binary's expected \
             version: {msg}"
        );
    }
}
