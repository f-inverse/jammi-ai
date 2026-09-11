# PRESSURE — pressure-test record (#500 plan)

## Round 1 (2026-09-10, on draft 9ccf2533) — two lenses, both REFINE

### Design lens (pressure-tester) — 10 block, 3 advisory; dispositions

| # | Finding | Disposition (where folded) |
|---|---|---|
| 1 | Default embedding loss (CoSENT, `trainer.rs:4356`), AnglE and MNRL are batch-coupled; per-rank gradient averaging is not the global-batch gradient; the trait cannot express the fix | **Gather rule**: `all_gather` added to the trait; every rank computes the identical global loss over the gathered batch; gather backward keeps local slots; adapter grads summed. DESIGN §1 cost paragraph re-derived. README r6; U4b acceptance (b) gather exactness |
| 2 | Rank-local, data-dependent control flow (divergence skip, 3-strikes, early stop, mining refresh) breaks lockstep | **Lockstep rule**: global-batch-index step boundary; `all_reduce_max_flags`; rank-0 validation decisions broadcast. README r7; U4b acceptance (d) |
| 3 | Reduce set can differ per rank (sparse `GradStore`) | Canonical `trainable_vars` order, zero-filled. README r7; U4b |
| 4 | Scaler over the whole table leaks validation and breaks refactor parity; split and `batches_per_epoch` over the train prefix | Split arithmetic from `data.rs:477-481` in the descriptor via `validation_fraction`; scaler streamed over the train prefix into `from_targets`, bit-identical. README r3/r4; DESIGN §2; U2b |
| 5 | Row-group alignment to W·B contradicts K7 on the TrainingSet descriptor | Alignment dropped; reader slices. DESIGN §2 |
| 6 | `RunRank`/`FetchPartition` are authenticated but unauthorized compute + SSRF primitives | **Authorization**: assignment carries `(job_id, tenant, attempt, coordinator_worker_id)`; peer verifies running/claimed_by/lease via read-only catalog; ids not URLs; typed refusal. README r9; DESIGN §4; U5a acceptance (b) |
| 7 | GPU byte-equality unsupported (flash `atomicAdd`, NCCL channel split) | Byte oracles on hermetic legs only; GPU legs digest pair + tolerance until S5. README r16; DESIGN §6 |
| 8 | K7 table incomplete (`use_rslora`, `rank_pattern`, …); test-of-table is green-and-wrong | Canonical whole-spec serialization; exhaustive destructuring test. README r5; DESIGN §3; U3 (b) |
| 9 | GradCache and hard-negative mining are whole-set arms unmentioned | W=1-only with typed refusal; residency exemption; files in U2b scope. README r2; DESIGN §2, §8; U4a refusals |
| 10 | No peer attempt fence (split brain) | Fence on `attempt`, keyed `(job_id, rank)`. README r8; DESIGN §4; U5a (c), U5b (c) |
| A11 | Dropout/bucketing make W-invariance non-fp even for decomposable losses | Oracle at `lora_dropout=0`, rung pinned, ε per leg. DESIGN §6 |
| A12 | Cache-reuse handle and nullable 029 columns | Own name → reused prefix; reference-counted reaping; NULL never matches. README r5; DESIGN §3 |
| A13 | `ModelCache` key collides with plan 65's scope | Shared `CacheKey` shape. README r14; U4a |

### Sizing lens (pressure-tester) — 12 block, 3 advisory; dispositions

| # | Finding | Disposition |
|---|---|---|
| 1 | U2 refactor parity unsatisfiable (scaler) | Same fix as design #4 |
| 2 | Mining/GradCache absent from scope | Same as design #9; `hard_negative_miner.rs`, `gradcache.rs` in U2b |
| 3 | U2 separable on the producer/consumer seam | U2a / U2b |
| 4 | U5 separable on the service/coordinator seam; U6 binds to U5a | U5a / U5b; U6 ∥ U5b |
| 5 | Identity-before-gang justified wrongly; U4 splittable | U4a (trait, session, refusals; concurrent with U2a) / U4b; reason restated as double-write avoidance |
| 6 | PR-boundary reasons refuted | Honest reasons restated; PR-D optional fold recorded |
| 7 | Commit order inverts lane dependency | U7a is PR-B commit 1, U7b PR-C commit 1; artifacts land as lane follow-up commits |
| 8 | `runpod_lib.sh` `gpuCount: 1` hardcode; no cluster primitive | In U7a/U7b scope; U7a → L; S4 |
| 9 | table-providers pin misplaced/mis-sized (db-owned 0.10.1) | U1 scope corrected; compat types named; S3 sizes |
| 10 | U1 acceptance tautological and blind to db features | New ci.yml clippy step for `postgres,mysql` registered with the closure gate; version test is a guard |
| 11 | Spikes under-inclusive | S3, S4 added (S5 from design #7) |
| 12 | U3 completeness test asserts fields U4 introduces; hidden `manifest.rs` overlap | U3 tests fields at its commit; U4b extends; co-ownership recorded |
| A13 | U7 acceptance not RED-able | Schema test + P1 rules + guard wiring; provisioning → S4 |
| A14 | U8 three-process arm is distributed, not hermetic | Reclassified; harness in scope |
| A15 | "required green before merge" has no trigger | Manual dispatch wording, as U1 |

Verified by the lead before folding: `trainer.rs:4356` (CoSENT default), `:3862` (pairwise
(n,n)), `:4110` (MNRL (n,n)), `:2560-2567` (divergence skip), `:802`/`:824-836` (train-split
scaler), `:1120`/`:1616-1626` (mining, GradCache), `data.rs:477-481` (split arithmetic),
`regression_loss.rs:169` (`from_targets`), `crates/jammi-wire/src/fine_tune.rs:237-272`
(`FineTuneConfig` fields), `docs/plans/65-resolve-witness/README.md:23-26` (cache rekey),
`flash_bwd_kernel.h:122-123` (deterministic path), `crates/jammi-db/Cargo.toml:55` (0.10.1).

## Round 2 (2026-09-10, on v2 fd543451) — two lenses, both REFINE; every disposition in v3

### Design lens — 8 block, 3 advisory

| # | Finding | Disposition |
|---|---|---|
| 1 | "Sum, don't average" is exact only if no trainable parameter consumes gathered remote slots; classification applies the head inside the loss (`trainer.rs:2676-2679`) | Invariant stated; per-arm gather points (logits for classification; head output for regression; encoder outputs otherwise). README r6; DESIGN §4; U4b (b) adds classification + quantile regression |
| 2 | Refusal predicate `refresh_every > 0` is true by default (`fine_tune.rs:212-231`) — every W>1 job would be refused | Predicate is `hard_negatives.mine` or `cached`. README r2; DESIGN §2; U4a |
| 3 | Per-rank dropout RNG (`resume.rs:107`) unmodelled — resume of a gang not reproducible | Per-rank `dropout_positions` gathered to rank 0; seed `f(seed, rank)`; oracle across a resume. README r8; DESIGN §4, §6; U4b |
| 4 | Descriptor cannot hold `jammi-wire`/`jammi-ai` types (`jammi-db` depends on `jammi-numerics` only) | Opaque versioned canonical encoding; completeness test in the owning crate. README r5; DESIGN §3; U2a/U3 |
| 5 | `batches_per_epoch` ambiguous between B and W·B; LR horizon and trailing scale follow | `ceil(train_count / (W·B))`; every step quantity indexed by global batch; U2b lands it. README r3/r7; DESIGN §2 |
| 6 | Zero-row ranks in the trailing global batch unstated | Kept; 0-row tensors, zero counts; fixture with `train_count` not a multiple of W·B. README r3; DESIGN §2; U2b (b), U4b (b) |
| 7 | `Precomputed` arm splits by batch count (`data.rs:493-497`) | Tests-only arm stays outside the table path, unchanged. README r3; DESIGN §2; U2b |
| 8 | "Streams into the reduction" is not bit-identical (`from_targets` two-pass) | One collected `Vec<f32>`, `from_targets` once; named 4 B/row exemption. README r4; DESIGN §2; U2b |
| A9 | Fence keyed `(job_id, rank)` misses a rank that moved hosts; "lesser" vs "lesser or equal" | Fence on `job_id`; lesser-or-equal refused. README r8; DESIGN §4; U5a (c) |
| A10 | GPU ε chosen after the run | ε pre-registered per leg before the first gating run; digest pair never a failure until S5. README r16; DESIGN §6; U7a schema |
| A11 | Seven governance stems, not two; `CacheKey` `None` semantics; `in_flight` map | All seven quoted; `None` distinct; `in_flight` rekeyed. README r14, r23; DESIGN §4 |

### Sizing lens — 6 block, 8 advisory

| # | Finding | Disposition |
|---|---|---|
| 1 | U2a does not compile: `ResultTableKind` wire mirror (`jammi-wire/src/embedding.rs:95-116`, proto) | In U2a scope; wire-server owner; B5 |
| 2 | Doc-parity gate runs every PR; guide variant block was in U9 | Block lands with U2a and U3; U9 prose only |
| 3 | U6 (c) needs U5b's harness and matrix entry | Moved to U5b (e); U5b depends on U6 |
| 4 | Reachability registry is derived from every `ci/scripts` file; `runpod_gpu_gang.sh` reds it | Allowlist in U7a/U7b scope + acceptance |
| 5 | No order-determinism criterion in either half of U2 | U2a (d) ordered read-back on a >1-row-group fixture; U2b (e) streamed order == committed for `target_partitions ∈ {1,N}` |
| 6 | S1 never probes `all_gather` or `from_rank` | S1 extended; bound to U4a and U5b |
| A7 | U1's clippy step is a coverage lane, not RED at base | Labeled; version test is the oracle; base run recorded |
| A8 | `jammi-wire/build.rs` hand-lists protos | In U5a scope |
| A9 | Co-ownership map incomplete | Rows added (SIZING) |
| A10 | D1 contradicted by sequential crossings | D1 restated |
| A11 | PR bases unstated | Stated (SIZING, README hand-off) |
| A12 | U3, U6 = M; U4b = L understated | L, L, XL |
| A13 | GPU/gang test targets unnamed | Existing `gpu_capability` / `distributed` targets; no new `[[test]]` |
| A14 | S4 cost/approval and ledger path not on the README | Added; `runpod_lib.sh:1263` fixed |

Verified by the lead before folding: `trainer.rs:2676-2679` (classify inside loss), `:2245`
(regression head pre-loss), `fine_tune.rs:212-231` (`mine: false`, `refresh_every: 1`),
`:590-601` (`refresh_every == 0` refused when mining), `trainer.rs:1451` (`.mine` gate),
`data.rs:439-442`/`:493-497` (Precomputed tests-only, batch split), `cache.rs:43-45`
(`in_flight` keyed by id), `resume.rs:107`, `crates/jammi-db/Cargo.toml:44`,
`trainer.rs:843-852`, `jammi-wire/src/embedding.rs:95-101`, `embedding.proto:108-109`,
`check_doc_parity.py:127-136`, `execution_surface_reachability_allowlist.txt` (162 lines),
`jammi-wire/build.rs:22-33`.

## Round 3 (2026-09-10, on v3 db5dd701) — disposition check: REFINE, consistency only

Every round-1 and round-2 disposition was found present and consistent except three
consistency slips introduced by the v3 fold, plus citation slips. All were text edits with no
mechanism change; folded into v3.1 and verified by the lead by grep (no round 4):

| # | Finding | Disposition |
|---|---|---|
| 1 | U6 → U5b edge present in UNITS only; README table/hand-off and SIZING still said `U6 ∥ U5b` | README unit table, hand-off step 6, SIZING schedule/edges/alternative 6 corrected |
| 2 | Reachability allowlist missing from U7b scope/acceptance and the co-ownership row | Added to U7b; row extended |
| 3 | `TrainingCommon.world_size` in U4a without its construction sites, serde default, or the wire field | `#[serde(default)]` = 1 (D10); construction sites (`wire/training.rs:176`, `session.rs:1172`, `:1300`, `tests/it`) and `training.proto`/`jammi-wire/src/training.rs` in U4a with wire-server co-owner; README r17 |
| A4 | Citations: `fine_tune.rs:237-267` → `:237-445`; `runpod_lib.sh:1263` → `:1264`; `optimizer.rs:612` (correct as cited) | Corrected |
| A5 | U5b omitted `tests/distributed/main.rs`; U4b listed `store/manifest.rs` under ai-core | Corrected |
| A6 | U4a → U2b edge (the `world` argument) unstated | Stated in README table, U2b, SIZING edges |

Lead's grep verification of the fold (recorded here in place of a fourth round): every
"U6 ∥ U5b" removed; `U6` in U5b's dependency column; the allowlist string present under U7a
and U7b; `serde(default)` and `wire/training.rs` under U4a; `main.rs` under U5b; the three
citations corrected. Verdict of record: the plan is **PROCEED-equivalent** — the last standing
findings were consistency edits, verified closed.

## v4 (2026-09-10) — rebase on the jobs-fleet branch, 68 reconciliation, U8 as an extension unit

### Phase 0.5 (gap-analyzer on BRIEF-v4): invariant-crossing — B4, B5, B6, B1, B2, K5, K4, K7, K2, K6, K1, K3; 30 ambiguities

Every ambiguity is ruled in README rulings 26–46. Highlights: training kinds eligible for a gang
(r26); a peer is a fleet worker with a busy slot, no new worker state (r27); membership is 68's
(r28); three migrations (r30); the training set is not a partial result (r31); I-GANG authorization
on the peer listener with its own allowlist bucket (r34); released-vs-failed and the `releases`
counter (r36); the watchdog under the actuator rule (r37); `jammi-ballista` is a non-leaf crate, no
feature, publishable (r38); roles as listener knobs (r39); retries off globally (r40); one gang
mechanism (r41); `expire_dead_executors` as membership liveness (r42); the seam table and the one
upstream gap (r43); device pinning and bytes (r44); pre-swept names (r45); PR-C before PR-A (r46).
v3.1 rulings 9, 11 (tenant-scoped mount), 17 (`[training]` knobs), 18 (one migration) and 19's
ordering are superseded.

### Round 4 (on v4) — recorded below after the re-dispatch
