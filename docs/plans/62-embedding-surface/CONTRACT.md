# CONTRACT — embedding-surface unit (K4/K7), phase 2

Base: `b5bb7eb9` (main, post-M2). Binding plan: `PLAN.md` v1 + v2 deltas (v2 wins where they
amend). This contract adds per-domain `files_in_scope`, invariants, acceptance, and the exact
gates. One PR, commits on one branch (`feat/62-embedding-surface`). Authored by the lead
2026-08-28.

## Frame

Two constitution rows: **K7** (identity folds the COMPLETE output-affecting parameter set —
esc-057 is the live defect: pooling/tokenizer/weights mutate under a constant `model_id` with
no `DefinitionHash` change) and **K4** (remote byte-parity with embedded — the delta is DEVICE
coverage of the existing CPU bitwise assertion, transport-only). Refuted shapes are binding:
NO dispatch-counter serving assertion (fused arms are training-only by design; C15 forbids
counters on the wire), NO forced-arm encode A/B (ForcedFlash stays private, B1), NO
`attention_arm` in encode identity (post-hoc ≠ identity; provenance slot only), NO
MANIFEST_VERSION bump (the digest self-announces as a hash mismatch).

## E1 — jammi-db: model-content digest in `ModelIdentity` (domain: db)

- `files_in_scope`: `crates/jammi-db/src/store/manifest.rs` (ModelIdentity :209,
  ProducingDescriptor :282, definition_hash fold), `crates/jammi-db/tests/it/*` for identity
  tests. Shared-declaration rows (`lib.rs`/`Cargo.toml`/`error.rs`) only if strictly needed.
- Add a model-content digest determinant to `ModelIdentity` as `Option`-with-NullMeans (the
  external-producer import path has no local files to hash; `None` carries a typed reason,
  never a silent default). NO bespoke pooling field on the Embedding variant
  (manifest.rs:184-193 uniformity ruling — Inference collides identically otherwise). The
  digest folds into `definition_hash` for every variant that carries a `ModelIdentity`.
- NO MANIFEST_VERSION bump (v2 reshape 2: env not persisted; a bump hard-fails every
  pre-existing sidecar through result_digest_anchor into neighbor_graph/graph_propagation +
  the api_freeze frozen-format assert).
- Tests: hash-inequality (two identities differing only in digest → different
  `DefinitionHash`), determinism (`definition_hash_is_deterministic` family extended),
  None-vs-Some inequality, and the fold is exhaustive-by-type (compiler-checked match or
  struct-literal fold, no `..` elision on the folded struct).
- Invariants: K7, K1 (ProducingDescriptor untouched in shape → replay arms unaffected here;
  the ai-core commit threads the value), K5 (no migration touched).

## E2 — jammi-ai: digest computation + esc-057 red-green (domain: ai-core)

- `files_in_scope`: `crates/jammi-ai/src/model/backend/candle.rs` (load path :34-40 pooling
  read, :516-615 `pooling_from_config`), `crates/jammi-ai/src/model/mod.rs`, `crates/jammi-ai/src/pipeline/import.rs`
  (external-producer path → `None` + NullMeans reason), `crates/jammi-ai/src/pipeline/{embedding,recompute}.rs`,
  `crates/jammi-ai/src/session.rs`, `crates/jammi-ai/tests/it/{pooling_config,recompute,compute_precision}.rs`.
- Compute the digest ONCE per model load: sha256 of config + `1_Pooling/config.json` +
  tokenizer files (cheap), weights via the existing `sha256_and_len` helper. Thread it into
  the `ModelIdentity` built for embedding (and inference — uniformity) descriptors.
- **The esc-057 fix test (defect path, `closes_escape: esc-057`)**: mutate
  `1_Pooling/config.json` bytes under a constant `model_id` → `definition_hash` MUST differ;
  same for tokenizer bytes and weights bytes. Must be RED at base `b5bb7eb9` (fix-verifier
  will revert prod and require RED), GREEN on branch. Control: byte-identical model dirs →
  identical hash (determinism, non-vacuous control).
- replay/recompute: `replay_descriptor` (recompute.rs:212-229) threads the digest so a replay
  under a mutated model dir is REFUSED/mismatched, never silently different vectors under an
  identical hash.
- Invariants: K7, K1, K2 (digest IO errors are typed refusals, not silent None).
- Sequenced AFTER E1 — branch from db's E1 branch head, not from base.

## E3 — jammi-bench: `EncodeStepTier` (domain: bench)

- `files_in_scope`: `crates/jammi-bench/src/encode_step.rs` (new), `main.rs`/`report.rs`
  registration rows, tier tests.
- `IDENTITY_FIELDS` — comparison: seed, batch, seq, row_lengths, backbone_dtype /
  compute_precision, checkpoint shas+size, pooling, normalize, warmup, iters_measured.
  Provenance (NullMeans reasons, NEVER identity): device_name, kernels_disabled_*,
  flash_compiled, build_features, chunk_size, attention arm. `attention_arm` is FORBIDDEN in
  identity (v2 reshape 3: constant on the eval path = false determinant, forbidden by
  resolve_keys + the identity doctrine).
- Cardinality pin test on the Rust side (exact field count + names, so the Python mirror can
  pin against it).
- Invariants: K7 (identity completeness at the bench seam), B1/B2 (no consumer names).

## E4 — jammi-encoders: batch-composition invariance oracle (domain: numerics)

- `files_in_scope`: `crates/jammi-encoders/tests/` beside the M1b family (same-row encoded
  alone vs in-padded-batch, per reachable arm, anchored to same-composition f32 truth).
- The property is near-exact (MASKED_LOGIT -10000 underflows pad weights to exact zero in
  f32; only GEMM reduction grouping differs) — the bound MUST derive from a MEASURED
  same-composition floor (guide:109-119 discipline). Bound lands PROVISIONAL and
  CPU-hermetic tests must be green without hardware; the pod half finalizes it.
  Red controls (row_lengths off-by-one, window ±1) are CONJUNCTIVE and must separate above
  the measured floor on hardware, or the oracle is inadmissible and gets reshaped, never
  tuned. Fixture-regime discipline: any windowed-attention control asserts
  `segment_len >= half_window+2` in-test.
- Any new `gpu_capability` module carries its gpu-parity-cell marker + PENDING deletion in the
  same diff (check_gpu_parity_matrix reconciliation) or a stated no-marker reason.
- Invariants: K2, esc-045 floor discipline; NO dispatch-counter assertions (training-only arms).

## E5 — jammi-server: K4 transport-only device leg (domain: wire-server)

- `files_in_scope`: `crates/jammi-server/tests/it/grpc_remote_session.rs` (+ a GPU-gated it
  module if needed).
- The leg: ONE compute, two read paths — remote gRPC serve vs LOCAL READ-BACK OF THE SAME
  ARTIFACT, bitwise (`repeated float` is lossless), zero tolerance, per keyed row, on a
  head_dim-64 checkpoint, GPU-gated. CPU bitwise remote-vs-local is ALREADY asserted
  (grpc_remote_session.rs:174-176) — this is device coverage, cite it.
- `encode_query`'s two-compute form: RECORD the GPU repeat-determinism premise
  (GpuLane.deterministic) — recorded observation, promoted to a gate only if pod evidence
  holds. NO dispatch-counter delta assertion (deleted by v2 reshape 1; C15).
- Invariants: K4, C15 (counters never on wire), B5 (tenant scope untouched).

## E6 — ci/docs: identity-fields mirror, producer, gate restructure, ledger (domain: docs-ci)

- `files_in_scope`: `ci/scripts/perf/identity_fields.py` (+ `test_identity_fields_subset.py`),
  `ci/scripts/perf/encode_ab.sh` (new producer), `ci/scripts/perf/ab_merge.py` (premise-refusal
  reuse only — no weakening), `ci/scripts/check_cuda_run_artifacts.py` (restructure, OWN
  COMMIT), `.jammi/escapes.jsonl`, `docs/plans/62-embedding-surface/*`, `NOTICE` untouched
  (that's unit 63).
- `ENCODE_IDENTITY_FIELDS` mirrors E3's Rust list exactly; cardinality pins BOTH sides;
  producer `encode_ab.sh` with leg-premise refusal in the merger.
- `check_cuda_run_artifacts.py` `build_identity_tuples()` restructure (first-match block_re +
  hard-mapped file→tier) is a GATE EDIT: staged as its OWN commit, flagged SWARM_GATE_TOUCHED,
  human/admin-merge acknowledged in the PR body. Gate self-tests green.
- Ledger: esc-056 `eval_added → closed` (conditions met, recorded in the row); esc-057
  `open → eval_added` in the SAME commit that cites E2's landed fix test (the row keeps its
  symptom_spec verbatim).
- Docs: unit-62 records incl. C15 hand-off + C16 inheritance notes; KO-7 scan-root widening
  recorded as its OWN follow-up tightening PR (OQ7 ruling), not done here.

## Step 5 — pod evidence train (lead-run, artifacts land via a domain agent)

encode_ab shapes + 8-seed invariance sweep + E5's GPU leg + E4 floor/red-control separation,
per pod-validation discipline (push-stamp verified, no in-job filtering, verdict lines read,
per-(arch,build) determinism). Artifacts with ancestry-true shas; PROVISIONAL bounds
finalized from measured values; committed via bench/docs-ci with `--exact` producer
invocations satisfying `check_producer_provenance_gates` (A)/(B) and the cuda-run artifact
gate.

## Acceptance (the feature RED oracle, phase 6)

1. esc-057 red-green (fix-verifier): pooling/tokenizer/weights byte mutation under constant
   `model_id` changes `definition_hash`; RED at `b5bb7eb9`, GREEN on branch; determinism control.
2. Identity-audited encode legs exist end-to-end: EncodeStepTier + ENCODE_IDENTITY_FIELDS
   equality + cardinality pins both sides (RED at base: no encode tuple exists anywhere).
3. The invariance oracle exists, is floor-derived, and its red controls are proven RED on
   hardware before any bound gates.
4. The K4 device leg asserts transport-only bitwise equality on GPU (RED at base: no GPU
   byte-parity assertion exists in any serving/it lane).

## Standing clauses

Every agent: own worktree + unique `CARGO_TARGET_DIR`; run YOUR CARD'S exact full gate
per-step `$?`, no pipe-masking; SHAs not branch names in every artifact/citation; gate edits
human-merged; numbers MEASURED not derived; docs reflect current state (no journey markers);
verifiers end final message with the fenced json verdict block LAST; `class_enumeration`
required in every phase-4/5/6 verdict.
