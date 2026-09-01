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
//!   dtype bug breaks it. Verbs: `generate_text_embeddings` / `encode_text_query`
//!   over BERT, ModernBERT, and the OpenCLIP text tower; `generate_image_embeddings`
//!   / `encode_image_query` over the OpenCLIP vision tower;
//!   `generate_audio_embeddings` / `encode_audio_query` over HTSAT-CLAP;
//!   `infer` (`Classification`, `Ner`) over ModernBERT; and the context-predictor
//!   `predict` forward pass (over one trained predictor, served on each device).
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
//! - **P4 — bf16 inference is admitted on Ampere+.** A `compute_precision=BF16`
//!   session loads and encodes on `gpu.device=0` (sm_86), proving the runtime
//!   compute-capability gate's acquire→decide→admit wiring, and its embedding
//!   matches the f32 direction. This is the only place the gate's *admit* path
//!   runs — the CPU suite reaches only the non-cuda reject arm and the decision
//!   predicate in isolation. See `bf16_gpu_gate`.
//! - **P5 — GGUF/k-quant serving + QLoRA on GPU (issue #351).** CPU↔GPU embed
//!   parity over a programmatically-written GGUF-quantized fixture (a
//!   Q8_1-activation-quantization-specific cosine floor — CUDA's quantized
//!   matmul re-quantizes the activation, CPU's does not, so the plain P1
//!   `COSINE_FLOOR` does not apply); a same-checkpoint GGUF-on-GPU vs
//!   f32-on-GPU quantization-loss floor; a QLoRA (`FrozenBase::Quantized`)
//!   fine-tune learning smoke on GPU; the resolver's `estimated_memory`
//!   checked truthful against a real `nvidia-smi`-measured device memory
//!   delta; and a printed (unasserted) quantized-vs-f32 throughput baseline.
//!   See `gguf_quantized_gpu`.
//!
//! Conformal / RRF are pure-CPU numerics and are out of scope — there is no GPU
//! kernel to validate for them.
//!
//! `ci/scripts/check_gpu_parity_matrix.py` is the coverage-COMPLETENESS gate
//! over this suite: it enumerates every SHIPPED (encoder architecture × GPU-
//! dispatching `ModelTask` verb) cell and requires each to be COVERED (a
//! `//! gpu-parity-cell: <Arch> × <Verb>` marker in one of these modules),
//! STRUCTURALLY_EXCLUDED, or PENDING — so an untested cell cannot silently
//! hide a divergence the way ModernBert×Classification once did (esc-028).
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

mod bf16_gpu_gate;
mod capability_surface;
mod classification_parity;
mod clip_text_embeddings_parity;
mod embeddings_parity;
mod fine_tune_learns;
mod gguf_quantized_gpu;
mod graph_finetune_learns;
mod graph_propagation_parity;
mod htsat_audio_embeddings_parity;
mod modernbert_embeddings_parity;
mod ner_parity;
mod open_clip_vision_parity;
mod predictor_parity;
