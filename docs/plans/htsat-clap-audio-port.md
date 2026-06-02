# HTSAT-Swin CLAP Audio Tower — Implementation Specification

**Status:** ready to implement (greenfield port; no prior code to preserve).
**Scope:** `jammi-encoders` (new audio tower) + `jammi-ai` model backend (HF arch dispatch). Engine-only; no consumer concepts.
**Workflow:** PLAN → PRESSURE-TEST → IMPLEMENT → AUDIT & REMEDIATE → COMMIT & PUSH, on a branch off `main`, PR, watch CI to green, merge. Do not auto-merge.

---

## 1. Motivation — jammi cannot embed real audio today

A live deployment verify (jammi-server on EC2, encode over gRPC) surfaced that jammi's audio
embedding is **non-functional against any real model**:

- `encode_query(model_id="laion/clap-htsat-unfused", modality=AUDIO)` → `No safetensors weights found` (that repo is `pytorch_model.bin`-only).
- `encode_query(model_id="laion/clap-htsat-fused", ...)` (which *has* `model.safetensors`) → **`Unsupported model architecture 'clap'`**.

Root cause: **`jammi_encoders::ClapAudio` is a flat single-scale ViT** — the open_clip *vision*-ViT
layout (conv patch-embed → uniform `ResidualAttentionBlock` stack with fused QKV + QuickGelu +
full non-causal attention → learned absolute `positional_embedding` → mean-pool → single
`audio_projection`). **Real CLAP — both the HF `transformers` `ClapModel` and the LAION
`open_clip` checkpoints — uses an HTSAT Swin transformer** as its audio encoder. These are
**different networks**, not the same network in two key-naming conventions, so no tensor remap can
bridge them. `ClapAudio` matches **no public checkpoint**; it has only ever loaded jammi's own
synthetic `tests/fixtures/generate_tiny_clap.py` fixture. JA1's "CLAP encoder" is effectively a
placeholder, and `cookbook/recipes/audio_search`'s "point at a real CLAP" is unfulfillable.

This is a **generic engine gap** (passes the discipline test: CLAP is *the* standard open
audio-embedding model and the HF repo — `laion/clap-htsat-fused`, ~22M downloads — is how everyone
obtains it). The correct fix is to give jammi a real HTSAT-Swin audio tower, not to hand-convert a
checkpoint into jammi's bespoke layout (a band-aid that forces every future user to redo it).

## 2. Goal

Implement a **real HTSAT-Swin audio encoder** in `jammi-encoders` (Rust/Candle) that loads a stock
HuggingFace `transformers` `ClapModel` audio branch (`laion/clap-htsat-fused`), wire HF-`clap`-arch
detection in the `jammi-ai` Candle backend to dispatch to it, and **replace the flat-ViT
`ClapAudio` placeholder** (it models a fictional network and only serves a synthetic fixture —
keeping it violates "code describes reality").

### Non-goals / constraints
- **Engine-not-platform:** no raga-mentor / Carnatic / consumer vocabulary anywhere.
- **Right abstraction, no band-aid:** a real HTSAT tower; never a silent weight-shape coercion onto the ViT; no hardcoded model ids/paths; errors-as-values; tail-safe.
- Keep the existing text/vision encoders and the audio **front-end contract** intact (see §5).
- The text tower of CLAP is out of scope unless `encode_query`/`generate_embeddings` audio paths need it — jammi's audio embedding uses the **audio tower** + `audio_projection` only.

## 3. Architecture facts (the target)

HF `ClapModel` audio branch (confirmed from `laion/clap-htsat-fused` `config.json`):
`audio_config.model_type = "clap_audio_model"`, `architectures: ["ClapModel"]`,
`depths: [2,2,6,2]` (4 hierarchical stages), `num_attention_heads: [4,8,16,32]` (per stage),
`window_size: 8`, plus `patch_size`/`patch_stride`/`spec_size`/`hidden_size`/`mlp_ratio`.

The audio encoder is an **HTSAT Swin Transformer**:
- **patch embedding** of the log-mel spectrogram,
- **4 hierarchical stages**; between stages a **patch-merging** downsample (halves spatial, doubles channels),
- within a stage, Swin blocks alternating **window** and **shifted-window** multi-head self-attention,
- **relative-position-bias** tables per layer (indexed by a precomputed relative-coordinate table),
- per-block LayerNorms (pre-norm), MLP with the configured `mlp_ratio`,
- a final norm, pooling, and a **`audio_projection`** linear into the shared latent dim.

HF safetensors key layout (audio branch):
`audio_model.audio_encoder.layers.{stage}.blocks.{i}.{norm1,attn.{qkv,proj},norm2,mlp.{fc1,fc2}}`,
`...attn.relative_position_bias_table`, `...attn.relative_position_index`,
`audio_model.audio_encoder.layers.{stage}.downsample.{reduction,norm}`,
patch-embed + final-norm keys, and `audio_projection.{linear1,linear2}` (verify exact names from the real config/checkpoint during Phase 0).

**No in-Candle reference:** `candle-transformers` 0.9.2 has **no** CLAP / Swin / HTSAT module
(verified). So unlike jammi's BERT/CLIP ports (which mirror `candle-transformers` and parity-test
against it), this port has no Candle reference to adapt — it is ported from the **PyTorch source**
(`transformers` `ClapAudioModel` / LAION `open_clip` HTSAT) and parity-tested against **PyTorch-derived
golden activations** (see §6).

## 4. Integration seams in jammi (where this plugs in)

- **Audio front-end contract** (`crates/jammi-encoders/src/audio.rs`): audio encoders map a log-mel
  batch `[batch, n_mels, n_frames]` → pooled embeddings `[batch, embed_dim]`, and report the
  feature-extraction params the decode/resample/mel front-end needs (`n_mels`, `n_frames`,
  `sample_rate`, `n_fft`, `hop_length`) so the front-end stays **config-driven, not hardcoded**.
  The new tower must satisfy this and report the params matching HF `ClapFeatureExtractor`
  (so the spectrogram fed to the tower matches the reference).
- **Encoder dispatch** (`crates/jammi-ai/src/model/backend/candle.rs`): `CandleAudioForward` is
  implemented for `ClapAudio` (~line 100); `Text`/`Vision`/`Audio` forward traits dispatch by
  architecture. `Unsupported model architecture` is raised here. Detect the HF CLAP arch
  (`architectures: ["ClapModel"]` / `audio_config.model_type == "clap_audio_model"`) and dispatch to
  the new tower, **mirroring the existing BERT/CLIP HF-arch dispatch**.
- **Config parse**: add `ClapAudioConfig::from_hf_clap_config` (beside the existing
  `from_clap_config` in `crates/jammi-encoders/src/clap_audio.rs`) — or a new config type for the
  HTSAT tower — parsing the HF `audio_config` schema.
- **Model resolver** (`crates/jammi-ai/src/model/resolver.rs`): already downloads safetensors +
  config for the Candle backend; ensure the HF CLAP repo resolves (it has `model.safetensors`).

## 5. Implementation strategy — MANDATORY: golden-reference, bottom-up, per-block parity

The point is to make this **not all-or-nothing**: decompose into small units each verified
numerically against a golden reference dumped from the real PyTorch model, so any divergence is
localized to the unit just added. This extends jammi's existing parity discipline
(`crates/jammi-encoders/tests/parity.rs`, gated behind the `parity-test` feature) — the only new
thing is the oracle is **Python-derived committed goldens** rather than `candle-transformers`.

### Phase 0 — Oracle + harness FIRST (before any tower code)
1. **Python oracle script** (committed, e.g. `tests/fixtures/generate_htsat_clap.py`) using
   `transformers` `ClapModel`:
   - (a) generate a **tiny real-HTSAT fixture** — a small HF `ClapModel` (reduced depths/dims/window,
     real `audio_model.audio_encoder.*` + `audio_projection` key names) saved as `config.json` +
     `model.safetensors`;
   - (b) run a forward pass on a **pinned input** with **forward hooks at every boundary**
     (mel front-end out, patch-embed out, each Swin block out, each stage out, final norm, pooled,
     projected embedding) → save per-layer **golden activations** as `.safetensors`.
   - Also generate goldens for the **real `laion/clap-htsat-fused`** (final embedding + key boundary
     activations for the pinned clip) for the live test.
   - **Environment note:** this needs torch + transformers able to run `ClapModel`. **x86_64 macOS
     caps torch at 2.2.2** and cannot run a current `transformers` `ClapModel` — run the oracle on
     **Linux** (CI container, the EC2 box, or a GPU/Colab box), commit the resulting golden tensors.
     Tests assert against the committed goldens; **CI needs no torch**.
2. **Parity harness** (Rust test helper, gated behind a `parity-test`-style feature): load a named
   golden tensor, assert jammi's tensor matches within a **strict fp32 tolerance** (e.g. max-abs-diff
   < 1e-4). Mirror `tests/parity.rs`.
3. **Pin a deterministic input**: a fixed audio clip *and* its precomputed spectrogram tensor (so the
   tower is testable independent of the front-end).

### Phase 1 — Separate the three failure domains
- **Front-end vs tower:** the audio→log-mel front-end (`sample_rate`, `n_fft`, `hop_length`,
  `n_mels`, normalization — must match HF `ClapFeatureExtractor`) is its own parity unit vs the
  golden mel. Then test the tower on the **exact golden spectrogram tensor** — never conflate
  "mel wrong" with "tower wrong."
- **Math vs weights:** verify the forward pass with shared known weights before trusting real
  weights; then a **weight-key coverage test** — every safetensors tensor consumed exactly once,
  every expected key found, every shape checked, none missing/extra (catches silent mis-mapping).
- **Config:** parse the real `config.json`; assert every derived hyperparameter (stage depths,
  per-stage heads, window size, embed dim, patch/spec sizes, mel params) equals the reference,
  before any forward pass.

### Phase 2 — Bottom-up port, a parity gate at EVERY level (no composing on unverified units)
Implement + parity-pass each unit against its golden boundary before building the next:
LayerNorm/Linear → patch-embed → **windowed MSA + relative-position-bias** (dedicated check on the
bias index table) → **shifted-window** variant (cyclic shift + attention mask — dedicated check) →
patch-merging → one Swin block → one stage (blocks + downsample) → full tower → final norm + pool +
`audio_projection`. A failure localizes to the block just added. The shifted-window mask, the
relative-position-bias indexing, and the patch-merge ordering are the classic divergence points and
each gets its own parity assertion.

### Phase 3 — Acceptance (two independent proofs)
- **Numerical (live-gated):** load real `laion/clap-htsat-fused`, encode the pinned clip, assert the
  embedding matches the committed golden within tolerance.
- **Behavioral (hermetic where possible):** a semantic/retrieval sanity check — similar audio ranks
  closer than dissimilar — so a one-input numerical coincidence can't pass for correct.

## 6. Test tiers
- **Hermetic CI (every PR):** tiny-fixture per-block parity + front-end parity + config parity +
  weight-key coverage + behavioral-on-tiny. Fast, no network, **no torch**.
- **Live opt-in** (gated by the existing live flag, e.g. `live-hub-tests`): real
  `laion/clap-htsat-fused` download + numerical parity vs the committed golden + retrieval sanity.
  Compile-checked in CI; runs behind the flag.

## 7. Definition of Done
- A real HTSAT-Swin tower in `jammi-encoders`, loading `laion/clap-htsat-fused`'s `audio_model.audio_encoder.*` + `audio_projection` with full weight-key coverage.
- HF `clap`-arch dispatch wired in `candle.rs` (mirrors BERT/CLIP); `Unsupported model architecture 'clap'` no longer raised for a real CLAP repo.
- Flat-ViT `ClapAudio` placeholder **replaced** (and the synthetic `tiny_clap` open_clip fixture retired or migrated — state the decision).
- Per-block parity tests pass against goldens (hermetic tiny tier); live real-model parity + retrieval sanity pass behind the flag.
- COMPLETE gate green: `cargo check --workspace --tests` · `cargo clippy --workspace --all-targets -- -D warnings` · `cargo fmt --all --check` · `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude jammi-python --no-deps` · `cargo test -p jammi-encoders -p jammi-ai -p jammi-db -p jammi-server` · `cargo test -p jammi-ai --features wire`.

## 8. Risks & mitigations
- **Silent numerical divergence** (a wrong axis/scale/activation produces plausible-but-wrong vectors): mitigated by per-boundary golden parity with strict tolerance — the core methodology.
- **No Candle reference** (can't adapt proven code): port from the PyTorch source; the Python goldens are the safety net.
- **Swin detail traps** — shifted-window attention mask, relative-position-bias index table, patch-merge order: each gets a dedicated parity check; if a block won't reach parity, STOP there and report exactly where (do not loosen tolerance).
- **Front-end mismatch** (mel params off → embeddings differ even with a perfect tower): front-end is parity-checked vs HF `ClapFeatureExtractor` independently.
- **Oracle environment** (torch unavailable on Intel macOS): generate goldens on Linux; commit them; CI needs no torch.

## 9. References
- `crates/jammi-encoders/src/clap_audio.rs` — the flat-ViT placeholder to replace; `ClapAudioConfig`, `from_clap_config`, `load`.
- `crates/jammi-encoders/src/audio.rs` — the audio-encoder family enum + front-end param contract.
- `crates/jammi-encoders/src/bert.rs` + `crates/jammi-encoders/tests/parity.rs` — the porting + numerical-parity pattern to follow (here the oracle is Python goldens, not candle-transformers).
- `crates/jammi-ai/src/model/backend/candle.rs` — `CandleAudioForward`, the arch dispatch, where `Unsupported model architecture` is raised.
- `crates/jammi-ai/src/model/resolver.rs` — HF download + safetensors/config resolution.
- `tests/fixtures/generate_tiny_clap.py` — the existing (synthetic, flat-ViT) fixture generator to supersede.
- Reference impls: HF `transformers` `ClapAudioModel` (`modeling_clap.py`), [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) (open_clip HTSAT), [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) `config.json` + `model.safetensors`.
