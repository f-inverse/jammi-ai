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

## Round 2 (on v2) — recorded below by the lead after the re-dispatch
