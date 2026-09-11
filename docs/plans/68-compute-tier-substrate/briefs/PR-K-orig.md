# Brief PR-K — Kubernetes reference unit (#482 item 2): Shape C kustomize base, shape-d + ci overlays, per-PR kubeconform gate, kind smoke, docs

Closes #482 item 2. One PR, cut from main AFTER PR-C merges (precondition, non-negotiable — see "Config spelling"). Plan of record: `scratchpad/plans/PLAN-K.md` (Decisions, pin table, Commits 1–3, Refinements V1–V10; later wins). Lead decisions: `scratchpad/decisions-K8s.md`. Contracts: `scratchpad/contracts/K1-kustomize-base.md`, `K2-overlays.md`, `K3-kubeconform-gate.md`, `K4-kind-smoke.md` (4a python, 4b docs-ci), `K5-docs.md`.

## Precondition: cut after PR-C merges
The manifests' TOML uses `[server] services = ["core", "event", "eval"]` + `[worker] enabled = false` (query tier) and `services = []` + `[worker] enabled = true, kinds = ["fine_tune", "graph_fine_tune", "context_predictor"]` (compute tier). On today's main that is an unknown-section hard load error: main still spells `[training] run_worker` (`crates/jammi-db/src/config/mod.rs:1049`) and still has a `train` tier (`crates/jammi-server/src/tiers.rs:92`). PR-C (`feat/deploy-shapes-C-jobs`, wt-C@656ee35d) lands `WorkerConfig` (`wt-C config/mod.rs:226,1036-1066`), tiers core/event/eval only (`wt-C tiers.rs:88-90,117`), and the worker kinds (`wt-C crates/jammi-ai/src/fine_tune/worker.rs:167-169`). The guide's fence test proves the two TOML files under the real loader (K5), so a pre-PR-C cut would be red by construction.

## Standing principles
- philosophy.md:125: orchestration is the consumer's runtime. `deploy/kubernetes/base` is the whole shape the engine cares about; ingress/TLS/HPA/NetworkPolicy/per-cloud annotations are the deployer's overlay — the seam is a kustomize patch, never an engine knob. NO per-cloud overlays, NO Helm, NO operator (deviation from the issue's literal "Fix 2"; one README paragraph names the seam).
- Docs reflect current state; every literal in the page has a producer in the tree (the manifests are `{{#include}}`d, never re-typed).
- **The shape-d compute-tier overlay is PROVISIONAL** until #500 decides the gang primitive (lead's comment on #482): a plain Deployment today; a StatefulSet or indexed Job once ranks need stable identity and ordered startup. The Shape C base is unaffected. K2's file header, K2's README, and K5's page all state this with a link to https://github.com/f-inverse/jammi-ai/issues/500. Do not pre-empt #500 with a StatefulSet.

## Verified facts (2026-09-10, main@7561658e)
- `deploy/` holds `.env.example`, `docker-compose.yml`, `docker-compose.ci.yml`, `jammi.selfcontained.toml` — no `kubernetes/`. The guide's Shape C/D manifests are sketches (`reference-topologies.md:155-233,254-292`) with "The engine ships no Helm chart and no manifest tree" at `:160`. `deploy-server.md` repeats no such claim (no edit).
- The Secret keys are the three env vars `docker-compose.yml:30-33` sets. Image uid 65532, `/etc/jammi/jammi.toml` third in the resolution chain, `HF_HOME=/var/lib/jammi/hf` (`Dockerfile:153-166,187,202,204`; cu12 `:269-281,292,301`). Kubernetes needs no exec probe (`probe.rs:12-17`).
- `compose-smoke.yml` (the cadence and step shape to mirror) gained a `resolve-base` job and an arm64 matrix leg in E2/#497 (`:47-65,70-78,105-106`) AFTER PLAN-K was written — kube-smoke carries `resolve-base` and runs one amd64 leg (K4b).
- Gates: `check_no_consumer_names.py` scans `crates` only (`:66`) — manual grep over `deploy/kubernetes` and `tests/compose` recorded in the PR body; `SWARM_GATE_TOUCHED` trips only on `.claude/agents/*.md`, `ci/scripts/**/check_*.{py,sh}`, `.claude/{settings.json,evals,hooks}`, `swarm.yml` (`swarm.yml:99-105`) — this unit touches none; `check_ci_guard_wiring.py` tracks every tracked `tests/**/test_*.py` (`:218-244`) — the new suite is wired in the same commit; P6 exempts a literal `push: "false"` (`check_gpu_prove_once.py:994-1016`); the fence test pins 27 selected fences (`docs_config_fences.rs:233`) and reads raw markdown (does not see includes — V1 fixes that, count → 29).
- Python client: `describe_source(id)` (`clients/python/jammi/_embedded.py:250-252`) is the id-keyed verb; `list_sources()` returns dicts (V4's `'patents' in db.list_sources()` is corrected to `describe_source("patents") is not None` in K4a).

## Pinned (copied from PLAN-K; do not update)
`helm/kind-action@06c1ae10762d3b9c1644e7fe69596ae519e015a2 # v1.15.0`; `kubectl_version: v1.34.11`; node image `kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d`; kubeconform `v0.8.0` linux-amd64 sha256 `9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883`; `--kubernetes-version 1.34.11`; kustomize `kustomize/v5.8.1` linux-amd64 sha256 `029a7f0f4e1932c52a0476cf02a0fd855c0bb85694b82c338fc648dcb53a819d`. OPEN: V5's `yannh/kubernetes-json-schema` commit sha for `-schema-location` is NOT in PLAN-K's table — the lead resolves and records it before K3 cuts; implementers must not invent it.

## Asks (one commit each; contracts carry the exact files, RED-first oracles and gates)
K1 (docs-ci) `deploy/kubernetes/base/{kustomization,deployment,service}.yaml` + `jammi.toml` — the guide's Shape C exactly (replicas 3, uid 65532 + fsGroup, httpGet `/readyz` + `/healthz`, `envFrom` secretRef `optional: false`, ConfigMap at `/etc/jammi`, emptyDir, `:latest` + pin comment, replicas-precondition comment).
K2 (docs-ci) overlays `shape-d` (cu12 Deployment `jammi-server-compute`, `gpu-node-pool: "true"`, `nvidia.com/gpu: 1`, PROVISIONAL header, kubeconform-only) and `ci` (`namespace: jammi-ci`, `postgres:16` + `nats:2.10-alpine -js` StatefulSets, `images:` → `jammi-ai-server:pr`, `imagePullPolicy: Never`, hostPath fixtures, no `[storage]`), `kind-config.yaml.in`, README (Secret line, env-wins, boundary paragraph, local validation, `disableNameSuffixHash` sentence, "kubeconform never inspects ConfigMap.data").
K3 (docs-ci) `ci.yml` job `Guard (kubernetes manifests)`: fetch + `sha256sum -c` both binaries, `kustomize build | kubeconform --strict --summary --kubernetes-version 1.34.11 -schema-location <pinned> -cache …` over base/shape-d/ci; in `ci-summary.needs`. No new `check_*.py`.
K4a (python) `tests/compose/remote_smoke.py` (oracle: `run(target, health_url, *, restart, after_restart)`, constants, `wait_for_ready`, `durable_after_restart`, `shared_catalog_after_restart`), `shape_b_remote.py` → thin Compose driver (`--dry-run` byte-identical to main; `compose-smoke.yml` untouched), `shape_c_kube_remote.py` (port-forward + `rollout restart` strategy, no durability assertion), `test_remote_smoke.py` (argv byte-equivalence, callback identity, non-vacuous negative controls) + its Guard leg.
K4b (docs-ci) `kube-smoke.yml`: compose-smoke's `on:` + V2's path-filtered `pull_request`; `resolve-base`; `docker-publish` `push: "false"`/`load`; kind-action pinned; `kind load` server + postgres + nats; namespace then Secret imperatively with `::add-mask::`; kubeconform before apply; rollout status; ONE-hop identity (`containerStatuses[].imageID` == `sha256:<docker .Id>` or ∈ crictl `repoDigests`); driver; `psql … select count(*) from sources` ≥ 1; logs on failure; `kind delete cluster` always.
K5 (docs-ci) `reference-topologies.md`: both sketches → `{{#include}}` of the real files; delete "ships no Helm chart and no manifest tree" and every "sketch"; boundary sentence; provisionality paragraph with the #500 link; Shape C fence re-spelled; V10 names the oracle + both drivers; "what kube-smoke proves / does not". `docs_config_fences.rs` resolves single-line `{{#include}}` fences (vacuity → RED → GREEN transcript required), 27 → 29. `docs.yml` paths + `deploy/**`. CHANGELOG state-only bullet.

## Acceptance
1. `Guard (kubernetes manifests)` green for base, shape-d, ci; in `ci-summary`.
2. `compose-smoke.yml` byte-unchanged; `shape_b_remote.py --dry-run` byte-identical to main; `test_remote_smoke.py` green in Guard.
3. `check_gpu_prove_once.py`, `check_ci_guard_wiring.py`, `check_execution_surface_reachability.py`, `check_swarm_bijection.py`, `check_doc_parity.py`, `check_citations.py` green; no new `check_*.py`; SWARM_GATE_TOUCHED clean.
4. `docs_toml_fences_parse_under_the_real_loader` green at 29 with includes resolved; the `[wroker]` corruption is RED.
5. `mdbook build`/`test` green; rendered page has no unresolved include, no "sketch", no "ships no Helm", and carries the #500 provisionality.
6. Manual consumer-name grep over `deploy/kubernetes` and `tests/compose` empty (PR body), plus the note proposing the `SCAN_TREE_ROOTS` widening as a separate human-merged tightening.
7. The PR's own `kube-smoke` run green (V2) with the identity echo and the `sources` count in the log — recorded in the ledger before merge.

## Out of scope
Per-cloud overlays; Helm; Ingress/TLS/HPA/NetworkPolicy/ServiceMonitor; StatefulSet/indexed-Job compute tier (#500); widening `check_no_consumer_names.py` (human-merged follow-up).

## Process rules learned on A–F (apply verbatim)
- Config lives in `crates/jammi-db/src/config/`; there is no config under jammi-server.
- Commit without trailers; the lead amends. Stage only your own files; every shared-class touch (K3/K4a's `ci.yml`, K5's `docs.yml`) goes in `scope_amendments`.
- Re-derive every path:line against the branch HEAD before citing; the post-PR-C tree is neither main@7561658e nor wt-C@656ee35d (wt-C also has uncommitted edits to `config/mod.rs`).
- MAINTAINER-GUIDE citations ARE gated: CI `Guard (citation resolver)` runs `python3 ci/scripts/perf/check_citations.py`; run it before every commit.
- CHANGELOG is state-only; no journey markers anywhere in the guide.
- Gate scripts (`ci/scripts/**/check_*.py`), agent cards and `swarm.yml` are human-amend-only (SWARM_GATE_TOUCHED); this unit adds none and edits none.
- Every pinned SHA/digest/version is copied from PLAN-K; never refreshed by an implementer.
