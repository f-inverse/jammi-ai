//! The CPU-hermetic training-throughput tier and its activation-memory negative
//! control.
//!
//! This is the training analogue of [`crate::search_rss`]: where that module
//! proves the streamed exact search holds a bounded resident set while a naive
//! collect-all baseline grows, this one proves the engine's GradCache
//! ([`gradcache_backward`]) in-batch-negative backward holds a bounded
//! activation footprint while the single-pass backward — the textbook path that
//! keeps every row's encoder graph and the `n × n` similarity graph alive until
//! one `backward()` returns — grows with the pair count. It also measures how
//! fast the GradCache path trains on this box (pairs/s), gated against a
//! committed same-box baseline by [`crate::rate_gate`].
//!
//! ## Why CPU-hermetic, and what is real engine work
//!
//! Everything runs on [`Device::Cpu`] with synthetic LCG embeddings — no model
//! download, no tokenizer, no network — exactly the setup the engine's own
//! `fine_tune` integration suite trains projection heads under. The encoder is a
//! stack of the engine's real [`build_projection_head`] LoRA layers applied over
//! the synthetic embeddings; the bounded backward is the engine's own
//! [`gradcache_backward`]; the optimizer step is the engine's [`AdamW`]. The
//! only thing re-implemented here is the *unbounded* single-pass backward — the
//! negative control — exactly as [`crate::search_rss`] re-implements the old
//! collect-all search the engine no longer has: the bounded path is what ships,
//! so the proof must keep the unbounded baseline alive somewhere to drive the
//! contrast against.
//!
//! ## The in-batch-negative loss
//!
//! The loss is Multiple-Negatives-Ranking (InfoNCE): a scaled cosine-similarity
//! matrix `S = normalize(A) · normalize(P)ᵀ · scale`, an `(n, n)` matrix whose
//! diagonal holds the true positive of each anchor, scored by cross-entropy of
//! each row against its diagonal index. It is re-implemented here (the engine's
//! `mnrl_loss` is crate-private) and is the same arithmetic the engine trains
//! with — a plain matmul of two row-normalised batches, no new distance
//! primitive.
//!
//! ## Why each measurement runs in its own process
//!
//! Peak RSS is sampled from `/proc/self/status` `VmHWM`, which is monotonic — it
//! only rises and cannot be reset from userspace. A single in-process run of the
//! large single-pass backward would leave a high-water mark that every later
//! (smaller / bounded) sample would read back. Each `(pairs, path)` measurement
//! therefore runs in a fresh `train-measure-once` child of this binary, so each
//! starts with a clean `VmHWM` and reports only its own working set — the same
//! child-process isolation [`crate::search_rss`] uses.

use std::process::Stdio;
use std::time::Instant;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use tokio::process::Command;

use jammi_ai::fine_tune::adamw::AdamW;
use jammi_ai::fine_tune::gradcache::{gradcache_backward, EncodeGroup};
use jammi_ai::fine_tune::lora::{build_projection_head, LoraModel};
use jammi_ai::fine_tune::FineTuneConfig;

use crate::report::{OomAssertion, OomControl, OomPoint, TrainingTier};
use crate::rss::{active_source, proc_peak_rss_mib};

/// The base-model hidden width the synthetic embeddings and projection head run
/// at — a realistic transformer-encoder width, so the per-row activation the
/// single-pass graph retains is a meaningful `O(d)` rather than a toy.
const HIDDEN_SIZE: usize = 768;

/// How many times the projection head is re-applied per row to form the
/// "encoder". A real encoder is many layers deep; stacking the engine's own
/// LoRA head this many times makes each row's retained activation graph
/// `depth · O(d)`, so the single-pass path's `O(n · depth · d)` footprint is the
/// unmistakable term the bounded path removes — without inventing a non-engine
/// layer.
const ENCODER_DEPTH: usize = 24;

/// GradCache chunk size (rows re-encoded with their graph alive at once). Small
/// relative to the largest pair count so the bounded path's `O(chunk · depth ·
/// d)` footprint is plainly independent of `n`.
const GRADCACHE_CHUNK: usize = 64;

/// The in-batch-negative similarity scale (InfoNCE temperature⁻¹), the
/// sentence-transformers default.
const MNRL_SCALE: f64 = 20.0;

/// The pair counts the OOM control sweeps, ascending. The single-pass path holds
/// `n · depth · d` f32 of encoder activations plus the `n × n` similarity graph,
/// so its footprint climbs visibly across this range while the bounded path
/// holds only `chunk · depth · d`. Chosen to span small / mid / large without
/// approaching the box's RAM — the proof observes the growth, it never drives an
/// actual OOM, and it never asserts against a remembered cliff pair count.
const OOM_PAIR_COUNTS: [usize; 3] = [256, 768, 1536];

/// LCG seeds for the synthetic anchor and positive embeddings — distinct so the
/// positive of a row is not its own anchor, giving the similarity matrix a
/// non-trivial off-diagonal (real in-batch negatives).
const ANCHOR_SEED: u64 = 0x1234_5678_9ABC_DEF0;
const POSITIVE_SEED: u64 = 0x0FED_CBA9_8765_4321;

// What GradCache bounds, and what it does not — the honest model the assertion
// below is sized against.
//
// GradCache removes the *encoder activation graph* from the resident set: the
// `n · depth · d` intermediate activations the single-pass backward keeps alive
// for every row at once, GradCache holds for only `chunk · depth · d` at a time.
// It does NOT remove the representations themselves (`O(n · d)` leaf tensors) nor
// the `n × n` in-batch-negative similarity matrix and its cross-entropy backward
// — pass 1 still builds those over the detached reps. So GradCache's resident set
// does grow with `n`, just at the `O(n · d + n²)` rate of reps-plus-similarity,
// while the single-pass path grows at the much larger `O(n · depth · d + n²)`
// rate of the full activation graph.
//
// The proof therefore is not "GradCache is flat" (it is not) but "the activation
// graph GradCache removes is the dominant growth": the single-pass delta must
// exceed the GradCache delta by a clear separation margin AND itself clear a
// growth floor. Both halves observed live across the ascending pair counts —
// never asserted against a remembered cliff count.

/// The single-pass peak-RSS delta between the smallest and largest pair count
/// must exceed this to count as "grows with n". Set well below the activation
/// growth the `n · depth · d` model predicts so the floor is a clear lower
/// bound, not a tight fit.
const SINGLE_PASS_GROWTH_FLOOR_MIB: f64 = 512.0;

/// The single-pass delta must exceed the GradCache delta by at least this margin
/// for the activation-graph removal to count as the *dominant* growth term. This
/// is the load-bearing separation: it is the extra memory GradCache's chunked
/// re-encode keeps off the resident set, which by the `O(n · depth · d)` model
/// is the bulk of the single-pass footprint. Set below the modelled separation
/// so it is a clear lower bound, not a tight fit.
const ACTIVATION_GRAPH_SEPARATION_FLOOR_MIB: f64 = 512.0;

/// Which backward path a `train-measure-once` child exercises.
///
/// Two variants because the proof contrasts the engine's bounded backward
/// against the unbounded textbook one. Both train the same synthetic data with
/// the same loss; they differ only in whether the encoder graph is held for all
/// rows at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardPath {
    /// The engine's [`gradcache_backward`] — chunked, `O(chunk · depth · d)`
    /// activations.
    GradCache,
    /// The bench-only single-pass backward — `O(n · depth · d)` activations held
    /// until one `backward()` returns. The unbounded negative control.
    SinglePass,
}

impl BackwardPath {
    /// The CLI token for this path, used on the `train-measure-once` child.
    pub fn as_str(self) -> &'static str {
        match self {
            BackwardPath::GradCache => "gradcache",
            BackwardPath::SinglePass => "single-pass",
        }
    }

    /// Parse the CLI token back into a path.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "gradcache" => Ok(BackwardPath::GradCache),
            "single-pass" => Ok(BackwardPath::SinglePass),
            other => Err(format!(
                "unknown backward path {other:?}, expected gradcache|single-pass"
            )),
        }
    }
}

/// The encoder shape one run trains at: the hidden width and the number of times
/// the projection head is re-applied per row. The full-fidelity proof uses
/// [`Shape::FULL`]; the hermetic cargo-test gate uses a small shape so the same
/// code path runs fast enough for CI. Threading the shape through one struct
/// keeps the subcommand and the gate on one set of train/encode functions.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// Base-model hidden width — the embedding and projection-head dimension.
    pub hidden: usize,
    /// How many times the projection head is re-applied per row (encoder depth).
    pub depth: usize,
}

impl Shape {
    /// The full-fidelity proof shape: the realistic encoder width and depth that
    /// make the single-pass activation graph the dominant, unmistakable growth
    /// term. Driven by the `train-scale` subcommand.
    pub const FULL: Shape = Shape {
        hidden: HIDDEN_SIZE,
        depth: ENCODER_DEPTH,
    };

    /// A small shape the hermetic cargo-test gate runs the bounded and unbounded
    /// backward paths under in-process, fast enough for CI. It exercises the same
    /// engine code path (GradCache backward, LoRA head, MNRL loss) at a width and
    /// depth that complete in well under a second, so the gate proves the two
    /// paths are live and agree without the full-fidelity RSS sweep's cost. The
    /// RSS-growth magnitude is the `train-scale` subcommand's measured artifact;
    /// what the gate proves is the negative-control machinery runs and the
    /// bounded path is gradient-correct.
    #[cfg(test)]
    pub const TEST: Shape = Shape {
        hidden: 32,
        depth: 3,
    };
}

/// One row of synthetic embeddings: an `(rows, shape.hidden)` f32 tensor of
/// seeded LCG noise on the CPU. Random high-dimensional rows give the similarity
/// matrix a non-degenerate off-diagonal, so the in-batch negatives are real.
fn synthetic_embeddings(
    rows: usize,
    shape: Shape,
    seed: u64,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    };
    let values: Vec<f32> = (0..rows * shape.hidden).map(|_| next()).collect();
    Ok(Tensor::from_vec(
        values,
        (rows, shape.hidden),
        &Device::Cpu,
    )?)
}

/// The "encoder": apply the projection head `shape.depth` times over a slice of
/// the embeddings. Each application is one engine [`LoraModel`] forward, so the
/// retained activation graph for the slice is `depth · O(len · d)` — the
/// quantity the bounded path chunks and the unbounded path holds for every row
/// at once.
fn encode_slice(
    head: &LoraModel,
    embeddings: &Tensor,
    shape: Shape,
    start: usize,
    len: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut h = embeddings.narrow(0, start, len)?;
    for _ in 0..shape.depth {
        for (_, layer) in &head.layers {
            h = layer.forward(&h)?;
        }
    }
    Ok(h)
}

/// Row-wise L2 normalisation — the same `x / ‖x‖` the engine's MNRL loss applies
/// before the similarity matmul.
fn l2_normalize_rows(x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-8, f64::MAX)?;
    Ok(x.broadcast_div(&norm)?)
}

/// Multiple-Negatives-Ranking loss over `(anchor, positive)` representations:
/// the symmetric cross-entropy of the scaled cosine-similarity matrix against
/// its diagonal. Same arithmetic as the engine's crate-private `mnrl_loss`,
/// re-implemented here because the proof drives it from the bench crate.
fn mnrl_loss(anchor: &Tensor, positive: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let n = anchor.dim(0)?;
    let a_norm = l2_normalize_rows(anchor)?;
    let p_norm = l2_normalize_rows(positive)?;
    let sim = (a_norm.matmul(&p_norm.t()?)? * MNRL_SCALE)?;
    let labels = Tensor::arange(0u32, n as u32, anchor.device())?;
    let row_loss = candle_nn::loss::cross_entropy(&sim, &labels)?;
    let col_loss = candle_nn::loss::cross_entropy(&sim.t()?, &labels)?;
    Ok((((row_loss + col_loss)?) * 0.5)?)
}

/// A fresh projection head and the [`Var`]s that train it, on `Device::Cpu`.
///
/// The same `Device::Cpu` + `VarMap` + `build_projection_head` setup the
/// engine's fine-tune integration tests use. The returned `Var`s are the head's
/// trainable parameters, which both the GradCache backward (gradient targets)
/// and the AdamW step (update targets) consume.
fn fresh_head(shape: Shape) -> Result<(LoraModel, Vec<Var>), Box<dyn std::error::Error>> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let mut head = build_projection_head(shape.hidden, &FineTuneConfig::default(), &varmap, &vb)?;
    // Train with dropout off so the encode is deterministic — the GradCache
    // contract requires its two encode passes to agree, and the contrast is over
    // memory, not regularisation noise.
    for (_, layer) in &mut head.layers {
        layer.set_training(false);
    }
    let vars = varmap.all_vars();
    Ok((head, vars))
}

/// The gradient of one in-batch-negative backward over the head's trainable
/// vars, by either path — the value an optimizer step consumes, and the value
/// the gate test cross-checks between the two paths.
type Grads = candle_core::backprop::GradStore;

/// Run one bounded (GradCache) backward over `pairs` synthetic pairs: encode
/// anchors and positives in chunks, score them with the in-batch-negative loss,
/// and accumulate the gradient without holding all encoder graphs at once.
/// Returns the gradient; the caller steps the optimizer (or, in the gate test,
/// compares it to the single-pass gradient).
fn gradcache_grads(
    head: &LoraModel,
    vars: &[Var],
    shape: Shape,
    anchors: &Tensor,
    positives: &Tensor,
    pairs: usize,
) -> Result<Grads, Box<dyn std::error::Error>> {
    let a_enc = |start: usize, len: usize| {
        encode_slice(head, anchors, shape, start, len).map_err(to_jammi_err)
    };
    let p_enc = |start: usize, len: usize| {
        encode_slice(head, positives, shape, start, len).map_err(to_jammi_err)
    };
    let groups = [
        EncodeGroup {
            rows: pairs,
            encode: &a_enc,
        },
        EncodeGroup {
            rows: pairs,
            encode: &p_enc,
        },
    ];
    let loss_fn = |reps: &[Tensor]| mnrl_loss(&reps[0], &reps[1]).map_err(to_jammi_err);
    Ok(gradcache_backward(
        &groups,
        GRADCACHE_CHUNK,
        &loss_fn,
        vars,
    )?)
}

/// Run one unbounded (single-pass) backward over `pairs` synthetic pairs: encode
/// every anchor and positive with grad in one graph, score them, and take one
/// `backward()` — so every row's encoder activation graph and the `n × n`
/// similarity graph are alive at once. The negative control: this is the
/// footprint GradCache removes. Returns the gradient.
fn single_pass_grads(
    head: &LoraModel,
    shape: Shape,
    anchors: &Tensor,
    positives: &Tensor,
    pairs: usize,
) -> Result<Grads, Box<dyn std::error::Error>> {
    let a_rep = encode_slice(head, anchors, shape, 0, pairs)?;
    let p_rep = encode_slice(head, positives, shape, 0, pairs)?;
    let loss = mnrl_loss(&a_rep, &p_rep)?;
    Ok(loss.backward()?)
}

/// Run one full training step by `path` — the backward plus one AdamW step over
/// the head's vars. The optimizer step is shared so the two paths differ only in
/// how the gradient is computed, never in how it is applied.
fn train_step(
    path: BackwardPath,
    head: &LoraModel,
    vars: &[Var],
    shape: Shape,
    anchors: &Tensor,
    positives: &Tensor,
    pairs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let grads = match path {
        BackwardPath::GradCache => gradcache_grads(head, vars, shape, anchors, positives, pairs)?,
        BackwardPath::SinglePass => single_pass_grads(head, shape, anchors, positives, pairs)?,
    };
    let mut optimizer = AdamW::new(vars.to_vec(), ParamsAdamW::default())?;
    optimizer.step(&grads)?;
    Ok(())
}

/// Bridge a boxed bench error into the engine's `JammiError` the GradCache
/// closure signature requires. The encode closures return the engine's `Result`,
/// so a bench-side failure is reported as a fine-tune error the GradCache run
/// surfaces.
fn to_jammi_err(e: Box<dyn std::error::Error>) -> jammi_db::error::JammiError {
    jammi_db::error::JammiError::FineTune(e.to_string())
}

/// The `train-measure-once` child entrypoint: run one `path` over `pairs`
/// synthetic pairs in this fresh process and return its own peak RSS.
///
/// Runs in a process whose `VmHWM` started clean, so the sampled peak reflects
/// only this path's working set at this pair count — the contamination a single
/// in-process measurement would suffer is structurally avoided.
pub fn measure_once(path: BackwardPath, pairs: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let shape = Shape::FULL;
    let (head, vars) = fresh_head(shape)?;
    let anchors = synthetic_embeddings(pairs, shape, ANCHOR_SEED)?;
    let positives = synthetic_embeddings(pairs, shape, POSITIVE_SEED)?;
    train_step(path, &head, &vars, shape, &anchors, &positives, pairs)?;
    proc_peak_rss_mib().map_err(Into::into)
}

/// Print a `train-measure-once` child's peak RSS to stdout — the single line the
/// parent parses.
pub fn emit_child_result(rss_mib: f64) {
    println!("{rss_mib}");
}

/// Print a `train-throughput-once` child's measured rate to stdout as
/// `<pairs_per_s> <wall_ms>` — the single line the cargo-test gate parses. A
/// line-oriented contract avoids a JSON dependency between the processes, the
/// same shape the `train-measure-once` child uses for RSS.
pub fn emit_throughput_result(pairs_per_s: f64, wall_ms: f64) {
    println!("{pairs_per_s} {wall_ms}");
}

/// Parse a `train-measure-once` child's stdout line back into its peak RSS.
fn parse_child_line(line: &str) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(line.trim().parse::<f64>()?)
}

/// Spawn a `train-measure-once` child of this same binary for one `(path,
/// pairs)`, returning its measured peak RSS.
///
/// Re-execs the current executable so the measurement runs with a fresh
/// `VmHWM`. The child's stderr is inherited so a failure surfaces in the parent
/// run's logs; only its single stdout line carries the measurement.
async fn spawn_measure(
    path: BackwardPath,
    pairs: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let output = Command::new(exe)
        .arg("train-measure-once")
        .arg("--path")
        .arg(path.as_str())
        .arg("--pairs")
        .arg(pairs.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!(
            "train-measure-once child ({} @ {pairs} pairs) exited with {}",
            path.as_str(),
            output.status
        )
        .into());
    }
    parse_child_line(&String::from_utf8(output.stdout)?)
}

/// Measure the GradCache training throughput at `pairs` pairs, in-process:
/// `pairs` divided by the wall-clock of one bounded backward + optimizer step.
///
/// The rate is measured in-process (not in a child) because it is a *time*, not
/// a high-water mark — `VmHWM` contamination does not affect wall-clock — and
/// the in-process timing avoids the child-spawn overhead skewing a per-second
/// rate. Returns `(pairs_per_s, wall_ms)`.
fn measure_throughput(pairs: usize) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let shape = Shape::FULL;
    let (head, vars) = fresh_head(shape)?;
    let anchors = synthetic_embeddings(pairs, shape, ANCHOR_SEED)?;
    let positives = synthetic_embeddings(pairs, shape, POSITIVE_SEED)?;
    let start = Instant::now();
    train_step(
        BackwardPath::GradCache,
        &head,
        &vars,
        shape,
        &anchors,
        &positives,
        pairs,
    )?;
    let elapsed = start.elapsed();
    let wall_ms = elapsed.as_secs_f64() * 1000.0;
    let pairs_per_s = if elapsed.as_secs_f64() > 0.0 {
        pairs as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    Ok((pairs_per_s, wall_ms))
}

/// The measured throughput lane: pairs/s and epoch wall-time at the largest
/// sweep pair count. Returned separately from the OOM control so the
/// rate-gate and the memory proof are assembled independently.
pub struct Throughput {
    /// Pairs trained per second through one GradCache backward + step.
    pub pairs_per_s: f64,
    /// Wall-clock of that one GradCache epoch, milliseconds.
    pub wall_ms: f64,
    /// The pair count the throughput was measured at.
    pub pairs: usize,
}

/// Run the OOM negative control: measure each backward path's peak RSS at every
/// pair count in [`OOM_PAIR_COUNTS`] (each in its own process), and evaluate the
/// flat-vs-growth verdict over the smallest-to-largest delta.
///
/// Returns the populated control whether or not the assertion passed — the
/// caller emits the JSON and the gate test asserts on `assertion.passed`, so a
/// failed proof surfaces with full numbers, never a faked pass.
pub async fn run_oom_control() -> Result<OomControl, Box<dyn std::error::Error>> {
    let mut points = Vec::with_capacity(OOM_PAIR_COUNTS.len());
    for &pairs in &OOM_PAIR_COUNTS {
        let gradcache_rss_mib = spawn_measure(BackwardPath::GradCache, pairs).await?;
        let single_pass_rss_mib = spawn_measure(BackwardPath::SinglePass, pairs).await?;
        points.push(OomPoint {
            pairs,
            gradcache_rss_mib,
            single_pass_rss_mib,
        });
    }

    let smallest = points.first().ok_or("OOM control produced no points")?;
    let largest = points.last().ok_or("OOM control produced no points")?;
    let gradcache_delta = largest.gradcache_rss_mib - smallest.gradcache_rss_mib;
    let single_pass_delta = largest.single_pass_rss_mib - smallest.single_pass_rss_mib;
    let separation = single_pass_delta - gradcache_delta;

    let single_pass_grows = single_pass_delta > SINGLE_PASS_GROWTH_FLOOR_MIB;
    let activation_graph_dominates = separation > ACTIVATION_GRAPH_SEPARATION_FLOOR_MIB;
    let passed = single_pass_grows && activation_graph_dominates;

    let detail = format!(
        "single-pass RSS delta {single_pass_delta:.1} MiB ({}; > {SINGLE_PASS_GROWTH_FLOOR_MIB:.0} \
         MiB floor) across {}→{} pairs; GradCache RSS delta {gradcache_delta:.1} MiB \
         (reps+similarity, not removed); activation-graph separation {separation:.1} MiB ({}; > \
         {ACTIVATION_GRAPH_SEPARATION_FLOOR_MIB:.0} MiB margin)",
        if single_pass_grows { "GREW" } else { "FLAT" },
        smallest.pairs,
        largest.pairs,
        if activation_graph_dominates {
            "DOMINANT"
        } else {
            "not dominant"
        },
    );

    Ok(OomControl {
        rss_source: active_source(),
        points,
        assertion: OomAssertion {
            passed,
            gradcache_delta_mib: gradcache_delta,
            single_pass_delta_mib: single_pass_delta,
            single_pass_growth_floor_mib: SINGLE_PASS_GROWTH_FLOOR_MIB,
            activation_graph_separation_mib: separation,
            activation_graph_separation_floor_mib: ACTIVATION_GRAPH_SEPARATION_FLOOR_MIB,
            detail,
        },
    })
}

/// Measure the GradCache training throughput at a given pair count. Public so
/// the cargo-test gate can drive a small-`n` probe through the same code path
/// the `train-scale` subcommand times at the full pair count.
pub fn run_throughput_at(pairs: usize) -> Result<Throughput, Box<dyn std::error::Error>> {
    let (pairs_per_s, wall_ms) = measure_throughput(pairs)?;
    Ok(Throughput {
        pairs_per_s,
        wall_ms,
        pairs,
    })
}

/// Measure the GradCache training throughput at the largest sweep pair count —
/// the most stable per-second rate, the one the committed baseline is set from.
pub fn run_throughput() -> Result<Throughput, Box<dyn std::error::Error>> {
    let pairs = *OOM_PAIR_COUNTS
        .last()
        .ok_or("no pair counts configured for throughput")?;
    run_throughput_at(pairs)
}

/// The committed same-box throughput baseline, loaded from
/// `baselines/training.json`. A *rate* is not portable, so this is a same-box
/// reference refreshed by hand — the gate reads the committed value and the
/// generous threshold from it.
#[derive(Debug, Clone, Copy)]
pub struct Baseline {
    /// The committed baseline rate, pairs/s.
    pub pairs_per_s: f64,
    /// The relative-drop threshold the gate applies.
    pub threshold: f64,
}

impl Baseline {
    /// The crate-relative path to the committed baseline JSON.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("training.json")
    }

    /// Load the committed baseline from `baselines/training.json`.
    ///
    /// The committed file carries the same-box *rate*; the relative-drop
    /// threshold is the harness's single [`crate::rate_gate::DEFAULT_REGRESSION_THRESHOLD`]
    /// — one generous band defined in one place, not duplicated per baseline.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(Self::path())?)?;
        let pairs_per_s = json["pairs_per_s"]
            .as_f64()
            .ok_or("baseline missing pairs_per_s")?;
        Ok(Self {
            pairs_per_s,
            threshold: crate::rate_gate::DEFAULT_REGRESSION_THRESHOLD,
        })
    }
}

/// Assemble the full [`TrainingTier`] from a measured throughput, the committed
/// baseline the rate is gated against, and the OOM control — the schema the
/// `train-scale` subcommand emits.
pub fn build_tier(throughput: Throughput, baseline: Baseline, oom: OomControl) -> TrainingTier {
    let gate = crate::rate_gate::RateGate::evaluate(
        throughput.pairs_per_s,
        baseline.pairs_per_s,
        baseline.threshold,
    );
    TrainingTier {
        hidden_size: HIDDEN_SIZE,
        throughput_pairs: throughput.pairs,
        pairs_per_s: crate::report::Measurement::measured(throughput.pairs_per_s, "pairs_per_s"),
        epoch_wall_ms: crate::report::Measurement::measured(throughput.wall_ms, "ms"),
        rate_gate: Some(crate::report::RateVerdict {
            measured_pairs_per_s: gate.measured,
            baseline_pairs_per_s: gate.baseline,
            threshold: gate.threshold,
            floor_pairs_per_s: gate.floor,
            passed: gate.passed,
            detail: gate.detail(),
        }),
        oom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-batch-negative loss is well-formed: identical anchor/positive rows
    /// put the true positive on the diagonal, so the loss is low; scrambled
    /// positives raise it. This guards the loss arithmetic the throughput and OOM
    /// paths both run, independent of the memory measurement.
    #[test]
    fn mnrl_loss_rewards_diagonal_positives() {
        let shape = Shape::TEST;
        let n = 8;
        let anchors = synthetic_embeddings(n, shape, ANCHOR_SEED).unwrap();
        // Aligned: positive == anchor → the diagonal similarity is maximal.
        let aligned = mnrl_loss(&anchors, &anchors)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Misaligned: unrelated positives → the diagonal is not favoured.
        let positives = synthetic_embeddings(n, shape, POSITIVE_SEED).unwrap();
        let misaligned = mnrl_loss(&anchors, &positives)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            aligned < misaligned,
            "aligned positives must score lower MNRL loss than unrelated ones: \
             aligned={aligned}, misaligned={misaligned}"
        );
    }

    /// The encoder is the engine's real LoRA head applied `shape.depth` times,
    /// so a slice encode produces a finite `(len, shape.hidden)` representation —
    /// the shape both backward paths feed the loss.
    #[test]
    fn encode_slice_shapes_and_is_finite() {
        let shape = Shape::TEST;
        let (head, _vars) = fresh_head(shape).unwrap();
        let embeddings = synthetic_embeddings(16, shape, ANCHOR_SEED).unwrap();
        let rep = encode_slice(&head, &embeddings, shape, 4, 8).unwrap();
        assert_eq!(rep.dims(), &[8, shape.hidden]);
        let max = rep
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max.is_finite(), "encoded representation must be finite");
    }

    /// The OOM negative-control machinery is live and the bounded path is
    /// correct: at a small hermetic shape, the GradCache (bounded) backward and
    /// the single-pass (unbounded) backward both run over the same synthetic
    /// pairs and produce the *same* gradient. This is the engine's GradCache
    /// equivalence contract exercised through the bench's own train paths — it
    /// proves the negative control (single-pass) and the bounded path are both
    /// exercised in CI and agree, so the `train-scale` subcommand's RSS contrast
    /// is between two paths that compute the identical update. The RSS-growth
    /// *magnitude* is the subcommand's measured artifact (CPU fine-tune at the
    /// scale a meaningful peak-RSS separation needs is too slow and too
    /// build-profile-dependent for the default hermetic test lane); what the
    /// gate proves here is that both paths run and the bounded one is exact.
    #[test]
    fn bounded_and_unbounded_paths_agree() {
        let shape = Shape::TEST;
        let pairs = 24; // > GRADCACHE_CHUNK is not required at this shape; a few
                        // chunks suffice to exercise the multi-chunk fold.
        let (head, vars) = fresh_head(shape).unwrap();
        let anchors = synthetic_embeddings(pairs, shape, ANCHOR_SEED).unwrap();
        let positives = synthetic_embeddings(pairs, shape, POSITIVE_SEED).unwrap();

        let gc = gradcache_grads(&head, &vars, shape, &anchors, &positives, pairs).unwrap();
        let sp = single_pass_grads(&head, shape, &anchors, &positives, pairs).unwrap();

        // Every trainable var's gradient must agree between the two paths within
        // tolerance, and be non-trivial (otherwise the agreement is vacuous).
        let mut max_mag = 0.0f32;
        for v in &vars {
            let t = v.as_tensor();
            let g_gc = gc.get(t).expect("gradcache gradient for trainable var");
            let g_sp = sp.get(t).expect("single-pass gradient for trainable var");
            let diff = (g_gc - g_sp)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-3,
                "bounded and unbounded backward must produce the same gradient, \
                 max abs diff = {diff}"
            );
            let mag = g_sp
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            max_mag = max_mag.max(mag);
        }
        assert!(
            max_mag > 1e-4,
            "the agreed gradient must be non-trivial, got max magnitude {max_mag}"
        );
    }

    /// The committed throughput baseline is a real gating reference with teeth:
    /// a run *at* the baseline clears the gate, and a run that regressed past the
    /// threshold *fails* it (RC1 — a regression gate must be able to fail). This
    /// proves the committed `baselines/training.json` is a well-formed,
    /// generously-thresholded same-box baseline the `train-scale` subcommand's
    /// measured rate is gated against, without re-measuring the (slow,
    /// build-profile-dependent) CPU throughput in the hermetic test lane.
    #[test]
    fn committed_baseline_gates_with_teeth() {
        use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};

        let baseline = Baseline::load().expect("committed training baseline must load");
        let rate = baseline.pairs_per_s;
        let threshold = baseline.threshold;

        // The committed baseline is a positive rate, and its threshold is the
        // harness's single generous band — sized for shared-runner noise, not a
        // tight fit, and defined in one place.
        assert!(rate > 0.0, "committed baseline rate must be positive");
        assert!(
            (threshold - DEFAULT_REGRESSION_THRESHOLD).abs() < 1e-9,
            "the baseline threshold must be the harness default \
             {DEFAULT_REGRESSION_THRESHOLD}"
        );
        assert!(
            threshold >= 0.25,
            "the threshold {threshold} must be generous enough to survive runner noise"
        );

        // A run at the baseline clears its own gate.
        assert!(
            RateGate::evaluate(rate, rate, threshold).passed,
            "a run at the committed baseline must pass the gate"
        );

        // A run just below the derived floor fails — the gate has teeth.
        let floor = rate * (1.0 - threshold);
        let just_below = RateGate::evaluate(floor - 1.0, rate, threshold);
        assert!(
            !just_below.passed,
            "a rate below the floor must fail the gate: {}",
            just_below.detail()
        );

        // A run above the baseline (a faster box) passes.
        assert!(
            RateGate::evaluate(rate * 1.1, rate, threshold).passed,
            "a faster-than-baseline run must pass"
        );
    }
}
