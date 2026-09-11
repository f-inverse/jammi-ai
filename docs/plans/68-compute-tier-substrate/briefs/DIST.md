# Brief DIST — distributed data plane research spike: which substrate carries bulk inference and beyond-one-node retrieval

Unit: research spike → decision document + the first implementable unit. Read-only on the tree; web research REQUIRED (Ballista, datafusion-distributed, Arrow Flight patterns). Independent of PR-C.

## Framing (from the lead's discussion; validate every premise)
Three lines, two regimes each (online latency-bound vs batch throughput-bound):
- Training: batch only; fleet today; gang = #500 (OUT OF SCOPE — do not design it; do not pick a gang primitive).
- Inference: online = embed the query string in-process on the query tier (never behind a scheduler); batch = embed a corpus / refresh a delta over partitions.
- Retrieval: online = top-k + filters + federation joins in-process on a replica holding the segments in its local cache (main crates/jammi-db/src/storage/index_cache.rs; SegmentedIndex fan-out crates/jammi-db/src/index/segment.rs); beyond-one-node = an index larger than a node's memory, graph builds, eval, retrieval-heavy SQL.

## Candidate substrates
S1. The jobs fleet itself (already shipped on wt-C): bulk inference as N `embedding` shard jobs writing one segment each + a fan-in publishing the version manifest (composes with GRAPH + DELTA; zero new dependencies; coordination through the catalog; workers interchangeable). Establish whether this makes a query-plane substrate unnecessary for BATCH INFERENCE entirely.
S2. Apache DataFusion Ballista (scheduler + executors, stages at partition boundaries, Flight shuffles). Facts to establish from primary sources: current release and DataFusion version it tracks vs jammi's pin (main Cargo.toml:71 `datafusion = "52.3"`, arrow 57); the `submit_physical_plan` path; how a custom `ExecutionPlan` (jammi's `InferenceExec` crates/jammi-ai/src/operator/inference_exec.rs:161, `AnnSearchExec` ann_search_exec.rs:104) is shipped to executors (PhysicalExtensionCodec); whether executors can be GPU-scheduled at all; latency profile of scheduler round trips; deployment shape (separate binaries) vs B4 "one binary".
S3. datafusion-distributed (datafusion-contrib): in-process distribution with no scheduler binary; establish maturity, version coupling, custom-plan support, how it exchanges partitions (Flight?), and whether a jammi replica set could act as its worker set (each replica = one binary, config change → B4 fit).
S4. In-house scatter-gather over Arrow Flight between replicas (jammi already depends on arrow-flight/flight-sql): each replica owns a segment subset (affinity), a coordinator fans DoGet out and merges via the existing SegmentedIndex total order; rescoring stays local to the owning replica. Establish the cost of building/maintaining this vs S3.

## Decisions the spike must produce (each with the principle and the reference)
- Whether batch inference needs anything beyond S1. (Lead's hypothesis: no.)
- Which of S2/S3/S4 is the query-plane substrate for beyond-one-node retrieval and federated SQL, judged on: B4 one-binary fit; version coupling risk with DataFusion (state the upgrade cadence both projects follow); custom ExecutionPlan support; where rescoring/mask application runs; tenant-scope enforcement per partition (B5: the TenantScopeAnalyzerRule must apply on every executor, not only the coordinator — establish whether each candidate can guarantee that); online latency (is the substrate ever on the hot path? it must not be).
- Segment affinity for the online replica set when the index exceeds one node: is that an engine mechanism (a replica declares which segments it serves; the coordinator routes) or the deployer's runtime? Apply the discipline test and the five-knob rule; recommend.
- What the FIRST implementable unit is (smallest thing that proves the substrate on the real operators), with its acceptance, and what is explicitly deferred.

## Constraints
- The library is never less capable than the server: whatever substrate is chosen, the single-process path must produce byte-identical results to the distributed path for the same operation (K4 analogue) — state the oracle.
- The engine ships no scheduler control loop of its own; if S2 is chosen, the scheduler is a deployer-run process and the manifests name it as a seam, not a shipped artifact — argue whether that is compatible with B4 at all.
- Do not propose Ray/Ray Serve (Python-side collectives and serving; ruled out by B4 in #500's framing) except to record why.

## Output
A decision document (not a code plan) with: premises verified, per-candidate facts table with URLs, the decision matrix against the criteria above, the recommended substrate, the first unit's plan (commits/oracles/acceptance per COMMON.md), open questions the lead must take to the user.
