# Common ground for every brief (read first)

Trees: `main` = /Users/vijaychakilam/git/f-inverse/jammi-ai @ 7561658e. `wt-C` = /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/f46fc0f2-c66b-40a7-8810-d51255669b7c/scratchpad/wt-C (branch feat/deploy-shapes-C-jobs, unmerged, 26 commits ahead of main: the jobs table, JobService, [worker] config, lease keeper, [server.limits], OTLP). Anything about jobs/workers/limits must be cited from wt-C; everything else from main. Re-derive every path:line yourself; never copy a citation from a brief without opening it.

Principles (binding; resolve every fork by citing one of these):
- docs/guide/src/philosophy.md — engine of generic primitives, names no consumer; the discipline test ("would a user who has never heard of any particular consumer reach for this?"); the leak-guards (stream knows nothing about payload; mutable tables = CRUD through DML, no built-in transition log or automatic versioning; provenance channels merge, never write); ONE consumption verb `search`; one binary / pluggable backends / five knobs; everything else (LB, ingress, TLS, secrets, IAM, observability stack, orchestration, autoscaling) is the consumer's runtime; the library is never less capable than the server; the default deployment fits on a laptop; production is a configuration change.
- docs/swarm/CONSTITUTION.md — B1 discipline test, B2 one-way references, B3 search-only consumption, B4 one binary, B5 tenant scope generic, B6 atomic across the workspace; K1 replay complete over descriptors, K2 validate at every input edge, K3 standardize in data space, K4 embedded == remote byte-for-byte, K5 append-only monotonic migrations, K6 lockstep version, K7 identity folds the complete output-affecting parameter set.
- "Engine ships the actuator, never the control loop that pulls it" (crates/jammi-ai/src/pipeline/recompute.rs:29-35).
- Greenfield: nothing existing is a constraint if the right design needs a rebuild — but a rebuild is still one atomic PR (B6) and migrations stay append-only (K5).
- Docs reflect current state: no journey markers, no "added in PR #N".
- Gate scripts (ci/scripts/**/check_*.py), agent cards, swarm.yml, CONSTITUTION are human-amend-only; a plan may PROPOSE a tightening as a separate human-merged PR, never fold one in.

Required plan shape (write it so a pressure-tester can attack it):
1. Decisions — every fork the brief left open, resolved, each citing the principle and the code fact that decides it. Unresolvable forks go in "Open for the lead" with the two options and your recommendation.
2. Premises — every "why" claim with the path:line that proves it on the named tree.
3. Design — the mechanism: where every value lives, every reader of that value, every failure mode and its behaviour.
4. Commits — one per capability (not per crate), files_in_scope, RED-first oracle (what fails at base, the exact observable), gates crossed.
5. Invariants crossed (B*/K*) and how each is preserved.
6. Acceptance — falsifiable criteria the acceptance-verifier can prove RED at base, GREEN on branch.
7. Out of scope — explicit.
8. References — external sources used to validate a decision (URL + one line on what it established). Use WebSearch/WebFetch when a decision rests on how a mature system behaves; do not cite from memory.
Cite `path:line` everywhere. Length: as long as needed, no padding.
