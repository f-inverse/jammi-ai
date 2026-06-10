//! Gated GPU-capability suite: proves the embedded engine's ML *correctness* on
//! a real CUDA device, closing the gap that the ML verbs are otherwise only
//! correctness-tested on CPU (GPU has been smoke-tested for device init +
//! memory only). This validation gates the GPU-ML release: a GPU-ML package
//! whose ML correctness on GPU is unproven must not ship.
//!
//! The suite proves three properties, each over the engine's *real* fixtures
//! (the cookbook `tiny_bert` encoder, `patents.parquet`, and the synthetic
//! graph / meta-dataset fixtures the CPU suites already use):
//!
//! - **P1 — CPU↔GPU parity** for the verbs with a real GPU kernel. The *same*
//!   input runs through a `gpu.device=0` session and a `gpu.device=-1` (CPU)
//!   session against the *same* fixtures, and the outputs must agree within an
//!   explicit tolerance. Parity is the decisive proof: a wrong GPU kernel or a
//!   dtype bug breaks it. Verbs: `generate_text_embeddings`, `encode_text_query`,
//!   and the context-predictor `predict` forward pass (over one trained
//!   predictor, served on each device).
//!   `propagate_embeddings` (SGC / APPNP) is also exercised on both devices, but
//!   propagation has **no GPU kernel** — it is a deterministic CPU `f64` fold —
//!   so its test asserts *device-independence* (bit-identical output regardless
//!   of `gpu.device`), not GPU-kernel parity. See `graph_propagation_parity`.
//! - **P2 — `fine_tune` learns on GPU.** A tiny real LoRA run on `gpu.device=0`
//!   completes, its training loss decreases first→last epoch, and the resulting
//!   adapter changes embeddings vs the base model (the on-device training math
//!   actually works, not just that it ran).
//! - **P3 — `fine_tune_graph` learns on GPU.** The end-to-end declared-graph
//!   fine-tune runs on `gpu.device=0`, completes, and learns (loss decreases /
//!   adapter changes embeddings).
//!
//! Conformal / RRF are pure-CPU numerics and are out of scope — there is no GPU
//! kernel to validate for them.
//!
//! ## Gating
//!
//! The suite is **off by default**: it compiles and runs only under the
//! `live-gpu-tests` cargo feature, and a meaningful run *also* needs the `cuda`
//! feature and a visible GPU. Every test early-returns with a `tracing::warn`
//! skip (never `#[ignore]`) when the `cuda` feature is off or no CUDA device
//! opens, so the default `cargo test` lane is unaffected. The GPU sessions pin
//! `require_gpu = true`, so on a CUDA build with a real GPU a parity test that
//! reached `select_device` *must* have run on the GPU — a GPU-less build fails
//! fast at session construction rather than silently degrading to CPU and
//! faking parity.
//!
//! The live run is a GPU-host (A10G) gate:
//! `cargo test -p jammi-ai --features cuda,live-gpu-tests gpu_capability \
//!  -- --nocapture --test-threads=1`.

mod harness;

mod embeddings_parity;
mod fine_tune_learns;
mod graph_finetune_learns;
mod graph_propagation_parity;
mod predictor_parity;
