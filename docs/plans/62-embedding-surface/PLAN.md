# Embedding-surface unit plan v1 (K4/K7) — reconstruction + plan
(Authored by the Plan agent 2026-08-28 against the M2 train tip e497bf0; persisted by the lead. The agent's full text with citations lives in its task output; this file carries the operative content.)

## Reconstruction highlights
- K4 (CONSTITUTION:62): remote surface matches embedded path BYTE-FOR-BYTE. Anchor: grpc_remote_session.rs:114 (CPU-only today).
- K7 (CONSTITUTION:65): content-addressable identity folds the producer's COMPLETE output-affecting parameter set, per-variant completeness audited.

## The five gaps
- G1: no identity-audited bench legs on the encode surface — ModelInferenceTier/GpuInferenceTier carry NO IDENTITY_FIELDS; identity_fields.py has no encode tuple; no encode producer script or leg-premise refusal.
- G2: K7 per-variant audit of ProducingDescriptor::Embedding never run; KNOWN unfolded output-affecting params: (a) pooling strategy (read from 1_Pooling/config.json with SILENT mean fallback, candle.rs:34-40 — different bytes, same DefinitionHash: live defect class), (b) model content (bare id string vs the finetune side's checkpoint shas), (c) dispatched kernel arm / kernels_disabled / build_features.
- G3: K4 unproven where the fused path runs — GPU serving tests assert well-formedness only (never byte-parity) AND use head_dim-16 tiny fixtures so flash/block arms NEVER dispatch in any serving/it lane; no dispatch-counter assertion. The zero-dispatch-is-RED class, one level up.
- G4: no batch-composition/padding-invariance oracle on pooled embeddings (encoded-alone vs encoded-in-padded-batch — the seam search crosses).
- G5: KO-7's scan roots exclude gpu_capability/** and jammi-server it-GPU tests — their skips are invisible to unrun-is-RED.

## M2 interaction (verified)
memeff is training-only, unreferenced outside jammi-kernels — no encode effect. Its chunk_size NullMeans-provenance doctrine binds this unit's identity design (jammi-only knobs = provenance slots, never shared identity).

## Plan (five commits + pod train)
- PR-A: K7 per-variant completeness audit doc on the Embedding variant + fold resolved pooling into identity (manifest format version bump, versioned-forward invalidation); replay_descriptor K1 companion; hash-inequality + determinism + version-rejection tests. Model-content digests + arm-in-identity go to the pressure-tester first.
- PR-B: EncodeStepTier in crates/jammi-bench/src/encode_step.rs with IDENTITY_FIELDS (comparison: seed,batch,seq,row_lengths,backbone_dtype/compute_precision,checkpoint shas+size,pooling,normalize,attention_arm,warmup,iters_measured; provenance: device_name,kernels_disabled_*,flash_compiled,build_features,chunk_size NullMeans slot); ENCODE_IDENTITY_FIELDS in identity_fields.py + cardinality pins both sides; ab_merge premise-refusal reuse; encode_ab.sh producer; check_cuda_run_artifacts (tier,producer_kind) registry extension staged as its own commit (gate edit, human-merged).
- PR-C: pooled-embedding batch-composition invariance oracle beside the M1b family (same-row alone vs in-padded-batch, per arm, anchored to same-composition f32 truth; truth-relative mean ratio over the 8-seed convention; PROVISIONAL→artifact-derived); conjunctive red controls (lengths off-by-one, window ±1) proven RED on hardware.
- PR-D: K4 byte-parity GPU leg on a head_dim-64 checkpoint (remote gRPC == embedded local, bitwise, zero tolerance, per keyed row) + dispatch-counter delta assertion (fused==layers×batches, declined==0); encode_query vs persisted-row cross-check; KO-7 scan-root widening flagged to the human gate-owner.
- Step 5: pod evidence train (encode_ab shapes + 8-seed invariance sweep), artifacts with ancestry-true shas, PROVISIONAL bound finalization, docs/plans/62-embedding-surface records incl. C15 hand-off + C16 inheritance notes.

## Open questions for the pressure-tester (rule, don't menu)
1. Arm-in-identity vs provenance (ComputeDevice-fold argument vs memeff doctrine vs cache fragmentation).
2. Model-content digests in ModelIdentity: weights-hash cost vs config-sha-only.
3. Tokenizer identity: in or recorded-out.
4. torch_encode.py twin now or C16-style front-door record.
5. Invariance-bound derivability vs the esc-045 paired/sign fallback.
6. Manifest-version-bump cache invalidation acceptable or migration shim.
7. KO-7 scan widening: this unit or its own tightening PR.

---
# v2 deltas (pressure-test REFINE folded, 2026-08-28 — plan CONVERGED, reshaped)

CONFIRMED: G1/G4 true; G5 true-but-deliberate (KO-7 scope is a reviewed decision — widening is its own human-merged tightening PR, ruling OQ7). G2a is a LIVE K7 defect on main: no pooling determinant anywhere in the definition_hash fold (exhaustive by type); different-bytes half already proven by pooling_config.rs:150-192. Harm = false attestation (verify_materialization Match on a different-pooling artifact) + false replay (recompute.rs:212-229 yields different vectors under a byte-identical hash) — NOT a stale cache hit (embedding probe is honestly UnpinnedAtInstant) and NOT a silent fallback (candle.rs:456-511 logs/errors — the defect is identity, not silence). Escape row esc-057 recorded OPEN on the M2 train.

REFUTED / RESHAPED:
1. G3's cause: fused arms are TRAINING-ONLY BY DESIGN (modernbert.rs:2543 `if self.training`, contract v4 §2 "eval/serving stays eager"; the whole class enumerated). PR-D's dispatch-counter assertion DELETED (unsatisfiable; independently blocked by C15 counters-never-on-wire). PR-B's forced-arm encode A/B DELETED (ForcedFlash is a private test seam; exposing it = consumer-driven engine surface).
2. PR-A reshaped: fold a MODEL-CONTENT DIGEST into ModelIdentity (config + 1_Pooling + tokenizer shas cheap; weights sha once per model load via sha256_and_len; Option/NullMeans for import.rs's external-producer path) — subsumes pooling/tokenizer/weights in one determinant; NO bespoke pooling field on the Embedding variant (manifest.rs:184-193's uniformity ruling; Inference collides identically otherwise). NO MANIFEST_VERSION bump (env not persisted — the change self-announces as a hash mismatch; a bump hard-fails every pre-existing sidecar through result_digest_anchor into neighbor_graph/graph_propagation + the api_freeze frozen-format assert).
3. PR-B reshaped: EncodeStepTier WITHOUT attention_arm (constant on the eval path = false determinant, forbidden by resolve_keys + the identity doctrine); kernels_disabled_*/flash_compiled/build_features stay provenance with NullMeans reasons. The check_cuda_run_artifacts (tier,producer_kind) work is a RESTRUCTURE of build_identity_tuples() (first-match block_re + hard-mapped file→tier), human-merged gate edit, staged alone.
4. PR-D reshaped to the transport-only K4 leg: remote gRPC serve vs LOCAL READ-BACK OF THE SAME ARTIFACT (one compute, two paths — bitwise is sound; the wire is lossless `repeated float`); encode_query's two-compute form first RECORDS the GPU repeat-determinism premise (GpuLane.deterministic) and promotes to a gate only if pod evidence holds (gpu_inference.rs:38-40's standing claim respected). CPU bitwise remote-vs-local ALREADY asserted (grpc_remote_session.rs:174-176) — the delta is device coverage.
5. PR-C: the invariance property is near-exact (MASKED_LOGIT -10000 underflows pad weights to exact zero in f32; only GEMM reduction grouping differs) — the bound must derive from a MEASURED same-composition floor (guide:109-119), red controls must separate above that floor or the oracle is inadmissible and gets reshaped, never tuned.
6. New-module gate interaction: any new gpu_capability module carries its gpu-parity-cell marker + PENDING deletion in the same diff (check_gpu_parity_matrix reconciliation), or a stated no-marker reason.

OQ RULINGS: (1) arm = provenance NEVER identity (definition_of requires hash computable BEFORE compute — a dispatched arm is post-hoc; memoization soundness); (2) model-content digest YES in ModelIdentity (see reshape 2); (3) tokenizer in via the same digest; (4) torch_encode.py NOT now — C16-style front-door record (eval is single-arm; a twin compares implementations, needs the esc-045 campaign discipline); (5) measured-floor-first (reshape 5); (6) NO version bump (reshape 2); (7) KO-7 widening = its own tightening PR.

SIZING: ONE train, identity commits first (CPU-hermetic, independently green), pod half contingent on producing admissible evidence in its window.
STATUS: CONVERGED for implementation, sequenced after M2 completes.
