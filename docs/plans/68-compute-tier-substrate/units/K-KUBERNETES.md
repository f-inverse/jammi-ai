# Kubernetes reference manifests — kustomize base and overlays, schema validation, kind smoke

`deploy/kubernetes/` is a reference shape for running `jammi-server` on Kubernetes: a kustomize base
(Shape C, the multi-replica query tier), a `shape-d` overlay (the disaggregated compute tier) and a
`ci` overlay (the stack the kind smoke stands up). Every pull request validates all three with
`kustomize build | kubeconform --strict` against a pinned Kubernetes schema; the `kube-smoke`
workflow applies the `ci` overlay to a real `kind` cluster and drives it with a remote client. The
guide (`docs/guide/src/reference-topologies.md`) includes the manifest files themselves, so the
published page and the deployed shape cannot drift. Discussion: issue #482 (item 2) for the manifest
tree, #500 for the gang primitive that shaped the compute tier.

The boundary is the engine's deployment stance — topology is configuration, one binary serves every
deployment shape (`docs/guide/src/philosophy.md#how-it-deploys-one-binary-pluggable-backends`).
Every difference between tiers is a ConfigMap or an env var; nothing here changes the engine.
Orchestration policy — ingress, TLS, autoscaling, network policy, disruption budgets, per-cloud
annotations — is the deployer's runtime.

---

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **The base is the Shape C query tier: `[server] services = ["core", "event", "eval"]`, `[worker] enabled = false`.** Every replica accepts every job submission and claims none. | `["core","event","eval"]` enumerates the whole tier universe (`crates/jammi-server/src/tiers.rs::ServiceTier`), so it equals `"all"`. The synchronous verbs (embed, infer, search) run inline in the serving process and never touch the claim loop (`crates/jammi-server/src/grpc/embedding.rs::generate_embeddings`). Consequence to know: `enqueue` always writes `JobExecution::Queued` (`crates/jammi-ai/src/jobs.rs`), so on a base-only deployment a queued job waits until a worker-enabled process joins the same catalog — the `shape-d` overlay, or `JAMMI_WORKER__ENABLED=true` on the base. The engine default is the opposite (`WorkerConfig::default`: `enabled = true`, every kind — "an unconfigured deployment is a whole one"); the base opts out explicitly because its replicas are sized and graced as a query tier (D5, D12). |
| D2 | **`shape-d` adds two ConfigMaps and leaves the base's untouched**: `jammi-compute-config` (`jammi-compute.toml`) and `jammi-scheduler-config` (`jammi-scheduler.toml`). Both set `[server] services = []` and `[worker] enabled = true`; the compute pods claim `["fine_tune", "graph_fine_tune", "context_predictor"]`, the scheduler claims `["fine_tune", "graph_fine_tune"]`. | The query tier already runs `enabled = false`, so no replace is needed. `context_predictor` has no placed arm, so listing it on the CPU scheduler would train it there instead of on a device. `services = []` still mounts `core`. Consequence to know: as shipped no pool lists `neighbor_graph`, `propagate`, `asof_join`, `embedding` or `infer` (`crates/jammi-ai/src/fine_tune/worker.rs::COMPILED_KINDS`), so those kinds stay queued when submitted as jobs; a deployment that enqueues them widens a pool's `kinds` or adds a pool (D10). See §2.2 for the ConfigMap key form. |
| D3 | **The `ci` overlay replaces the base ConfigMap** (`behavior: replace`) with the base file minus `[storage]` and `[distributed]`; `[worker]` stays off. | Dropping a stanza needs a replace, not a merge. Without `[storage]`, results land under `artifact_dir` on the scratch `emptyDir`. The smoke's `GenerateEmbeddings` runs inline, so it needs no claim loop. |
| D4 | **Per-pod identity comes from the downward API, never from the shared ConfigMap.** The compute StatefulSet derives `JAMMI_SERVER__PEER_ADVERTISE` and `JAMMI_BALLISTA__EXECUTOR__ADVERTISE_HOST` from `POD_NAME`/`POD_NAMESPACE` (`fieldRef`). The manifests do not set `JAMMI_WORKER_ID`. | Every replica mounts one ConfigMap, and these values differ per pod. Kubernetes expands `$(VAR)` only for variables declared earlier in the same `env` list, so `POD_NAME`/`POD_NAMESPACE` stay ordered first. `JAMMI_WORKER_ID` is the non-unique `instances.label` that `ListWorkers` shows (`crates/jammi-ai/src/fine_tune/worker.rs::worker_label`); a deployer who wants pod names there adds one `env` entry with `fieldRef: metadata.name`. That is safe because the env layer ignores a single-segment `JAMMI_*` name that is not a top-level config field (`crates/jammi-db/src/config/env_map.rs`), and `env` overrides `envFrom` (Pod v1 API). |
| D5 | **Nothing is sized.** No CPU/memory `resources` block, no PodDisruptionBudget, PriorityClass, PVC, HPA, tolerations or spot manifest. The compute StatefulSet carries `resources.limits: {nvidia.com/gpu: 2}` — the device-plugin scheduling request, one device per `[worker] local_ranks` (`[gpu] devices = [0, 1]`), not sizing. | Orchestration and autoscaling are the deployer's runtime. One number cannot cover an idle query replica and a LoRA run, and a shipped number reads as engine sizing. The shipped QoS class is therefore BestEffort; Burstable needs requests, Guaranteed needs limits equal to requests, applied as a strategic-merge patch on container `jammi-server` in the deployer's overlay. PriorityClass is cluster-scoped and a missing class rejects the pod; a PDB is app-owner created; an HPA needs a metrics adapter. |
| D6 | **Images stay `:latest`** (`ghcr.io/f-inverse/jammi-ai-server`, `…-server-cu12`), with pin advice in a comment. | A reference manifest that pins a version goes stale on every release. How the tags move (`.github/workflows/server-image.yml`): both arms publish `type=semver,pattern={{version}}` under `docker/metadata-action` v5.10.0's default `flavor: latest=auto`, which emits `latest` on every non-prerelease `v*` tag; the raw `latest,enable={{is_default_branch}}` line is inert on a tag ref. The CPU `:latest` additionally moves when a maintainer dispatches the workflow on `main` (`merge-cpu-main`); the cu12 image publishes only from a `v*` tag. Published semver tags are unprefixed (`0.49.1`, `0.49`) because `{{version}}` strips the `v`; the manifests' comments, the README and the guide spell the pin `:vX.Y.Z`, a tag the registry does not carry. Image floor: see §2.3. |
| D7 | **Liveness is `httpGet /healthz`, `periodSeconds: 10`**, all other fields at the Kubernetes defaults (`failureThreshold: 3`, `timeoutSeconds: 1`); readiness is `httpGet /readyz`, `periodSeconds: 5`. No exec probe. | `/healthz` is 503 exactly when the lease keeper thread is dead or the claim loop task panicked; a stopped or aborted loop, a process with no loop, and a DRAIN in progress are all 200 (`crates/jammi-server/src/runtime.rs::LivenessReport`). `failureThreshold × periodSeconds` = 30 s is therefore the restart-rate knob: a liveness restart of a pod holding a job sends that job down the lease-expiry path, which consumes an attempt. Kubernetes probes HTTP itself, so `jammi-server probe` adds nothing here (`crates/jammi-server/src/probe.rs`). |
| D8 | **`kube-smoke` asserts runtime facts, each non-vacuous**: rollout readiness, one-hop image identity, `get_server_info().broker == "jet_stream"`, the `sources` row count in Postgres, and that the pod created by a `rollout restart` sees the source its predecessor registered. | See §4. A compile-time feature list or a `>= 0` count would pass with the broker or catalog URL deleted; each oracle fails when the thing it names is absent. |
| D10 | **`[worker] kinds` is the pool seam; a second pool is a second workload, not a second overlay in this tree.** | Overlapping kind lists are safe: a job is claimed exactly once (`crates/jammi-db/src/catalog/jobs_repo.rs::claim_next` — `FOR UPDATE SKIP LOCKED` on Postgres, the single serialised writer on SQLite). Disjoint lists partition the queue. A kind listed by no pool stays queued. Selectors are immutable and overlapping controllers fight, so a copied pool substitutes its name everywhere the name is carried: `metadata.name`, `spec.selector.matchLabels.app`, the template's `labels.app`, the `configMapGenerator` name, the volume's `configMap.name`, and for a StatefulSet also `serviceName`, the headless Service, and the DNS strings in `env`. |
| D11 | **The compute tier is a StatefulSet behind a headless Service, plus a single-replica CPU scheduler Deployment behind a plain Service.** | See §2.1. |
| D12 | **The base keeps the Kubernetes default grace (30 s); both `shape-d` workloads set `terminationGracePeriodSeconds: 600`.** | `jammi-server` has two shutdown modes: SIGTERM = DRAIN (the in-flight job finishes, the lease keeps renewing, every epoch bundle lands, exit 0) and SIGINT or `jammi-server release` = RELEASE (leases are handed back at once; a confirmed release costs no attempt because the cap compares `attempts - releases` against `MAX_ATTEMPTS = 3`). There is no engine-side timeout — `[server.limits] request_timeout_secs` defaults to `None` — so the pod's grace is the only bound on a DRAIN. The operative rule: the grace must cover one epoch's wall time, or SIGKILL lands before the final bundle and the job takes the expiry path (one `[lease] duration_secs` window, one attempt); the successor resumes from `{job_id}/_resume/`. 600 s is the cluster-autoscaler's `--max-graceful-termination-sec` default, the largest value a scale-down honours; a larger one is honoured only by rollouts and `kubectl delete`. The query replicas hold no training job, so the default suffices there. The mechanism is `OPS-COMPUTE-TIER-OPERABILITY.md`; the rollout arithmetic and the `preStop` recipes are in `deploy/kubernetes/README.md` ("Shutdown: DRAIN and RELEASE"). |
| D13 | **The kubeconform schema source is pinned to a commit**: `yannh/kubernetes-json-schema@b582a12a09aa9b5d1edad577a964c504130214fc`, `--kubernetes-version 1.34.11`. | A moving schema branch makes the gate's result depend on the day it runs. The sha is load-bearing: altering one hex digit makes every resource fail with `could not find schema` (exit 1). |
| D14 | **The guide's config-fence test resolves `{{#include}}` and pins 30 selected fences**: 27 written in the guide plus `base/jammi.toml`, `shape-d/jammi-compute.toml`, `shape-d/jammi-scheduler.toml`. The `ci` overlay's TOML is not included by the guide and is not counted. | kubeconform never inspects `ConfigMap.data`, so the TOML bodies are proven by the real config loader instead (`crates/jammi-db/tests/it/docs_config_fences.rs::docs_toml_fences_parse_under_the_real_loader`). Without include resolution a fence whose body is one `{{#include …}}` line is invisible to the test. The `ci` TOML is proven by `kube-smoke`: a file that fails to load never becomes ready. |
| D15 | **No per-cloud overlays, no Helm chart, no operator, no Ingress/TLS/HPA/NetworkPolicy/ServiceMonitor.** The README names the seam: a kustomize patch on the Deployment/Service metadata, never a change to the engine's knobs; transport encryption is `docs/guide/src/security.md#transport-encryption-is-the-deployers-runtime-not-the-engines`. | Issue #482 proposed per-cloud overlays; they are consumer-side policy with a cloud's name on them. `base/` is the whole shape the engine cares about. |
| D16 | **The tree names no consumer and no cloud.** The GPU node label is the neutral `gpu-node-pool` ("your cluster's own GPU node label"), the sidecar images are upstream `postgres`/`nats`, and the S3 bucket and region are placeholders the deployer edits. | Jammi names no consumer anywhere. The consumer-name gate's structural leg scans `crates/` (`ci/scripts/check_no_consumer_names.py::SCAN_TREE_ROOTS`); its denylist leg covers every tracked file, `deploy/` included. |
| D17 | **The per-PR gate is direct `kustomize build \| kubeconform` steps in the `kube-manifests` job of `ci.yml`** (`Guard (kubernetes manifests)`, in `ci-summary.needs`), not a guard script. `kube-smoke` runs on push to `main`, nightly, on dispatch, and on pull requests touching `deploy/kubernetes/**`, `tests/compose/**`, `Dockerfile` or the workflow itself. One amd64 leg. | The gate is two upstream binaries applied directly and needs fetched tarballs plus a schema cache, which a hermetic guard in `ci/guards.toml` cannot carry. A new workflow file cannot be dispatched against its own PR branch, so the path-filtered `pull_request` trigger is its pre-merge proof, and it stays because a manifest, driver or Dockerfile change is the class it exists to catch. The pinned tarballs and `kind-action`'s kind binary are linux-amd64; the arm64 image is proven by `compose-smoke.yml`'s arm leg. `kube-smoke` reports to no `ci-summary`. |
| D18 | **The smoke script is one importable oracle plus two thin drivers**, the seam being a `restart` callback and an `after_restart` callback. | The Compose and Kubernetes shapes differ only in how the server is bounced and in what must hold afterwards; everything else is one body (§5). |
| D19 | **`generatorOptions: {disableNameSuffixHash: true}` in every kustomization.** | The workloads reference their ConfigMaps by static name; a hash suffix would break the reference. |

---

## 2. Design — where every value lives, who reads it, how it fails

**Config values.** Non-secret knobs live in one TOML per role, shipped as a ConfigMap and mounted
read-only under `/etc/jammi`. Secrets (`JAMMI_AUDIT_MASTER_KEY`, `JAMMI_CATALOG__POSTGRES__URL`,
`JAMMI_BROKER__JET_STREAM__URL`) live only in the Secret `jammi-server-secrets`, injected through
`envFrom` with `optional: false`; no Secret manifest is in git, and the README gives the one
imperative `kubectl create secret generic` line. The env layer (`JAMMI_<PATH>`) resolves after the
file, so an operator overrides any knob without editing a ConfigMap. The S3 driver reads its
credentials from the process environment, so `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` join the
same Secret when the result root is S3. The reader is `JammiConfig::load`, whose resolution chain
is: explicit `--config`, `JAMMI_CONFIG`, `./jammi.toml`, `/etc/jammi/jammi.toml`, the platform
config dir (`crates/jammi-db/src/config/mod.rs::resolve_config_path_in`). `JammiConfig` rejects
unknown keys, so an unknown section is a load error, never a silent default.

**Pods.** All three workloads run `runAsNonRoot`, `runAsUser: 65532`, `fsGroup: 65532` — the images'
nonroot uid, with `fsGroup` so the scratch `emptyDir` is writable without relying on kubelet's
default volume mode. The `emptyDir` at `/var/lib/jammi` holds `JAMMI_ARTIFACT_DIR`, `HF_HOME` and,
on cu12, `CUDA_CACHE_PATH` (`Dockerfile`); it is local scratch — the catalog, the broker and the
result root carry the durable state. Ports are `flight` 8081 and `health` 8080, plus `peer` 9000 on
both compute-tier roles, `ballista-sched` 50050 on the scheduler and `ballista-flt`/`ballista-grpc`
50051/50052 on the compute pods. The base Service has no `type:` and is therefore ClusterIP;
`replicas: 3` is safe because every replica runs the identical config against one Postgres catalog
and the lease plus advisory-lock mechanism holds across the fleet
(`docs/guide/src/catalog-and-broker.md#multi-writer-safety`).

**Tiers.** Base: query replicas, `[distributed] max_world_size = 2` read at the submit edge only.
`shape-d`: base unchanged, plus the compute StatefulSet and the scheduler Deployment. `ci`: base
minus `[storage]`/`[distributed]`, `replicas: 1`, `namespace: jammi-ci`, the image rewritten to the
kind-loaded `jammi-ai-server:pr` with `imagePullPolicy: Never`, hostPath fixtures, and upstream
`postgres:16` / `nats:2.10-alpine` StatefulSets each bound to its Service by `serviceName` (the
1.34 strict schema does not require the field, so only `kubectl apply` would catch its absence).

### 2.1 The compute tier

`jammi-server-compute` is a StatefulSet (`replicas: 2`, cu12 image, `nodeSelector
gpu-node-pool: "true"`) because each rank's peer address must stay stable across a pod restart for
gang membership (`instances.peer_addr`, `[server] peer_advertise`); a Deployment's pod names churn on
every reschedule and cannot be that identity. The headless Service is the DNS half of the same fact:
ranks dial `<pod>.jammi-server-compute.<namespace>.svc.cluster.local:9000` directly, never a virtual
IP. It sets `publishNotReadyAddresses: true` because the peer listener serves as soon as the server
binds, before `/readyz` reports ready (readiness also gates model preload); a coordinator dials a
peer only after its own readiness, but the peer it dials may still be warming, and without the flag
a warming pod has no DNS record. `podManagementPolicy: Parallel` holds because ranks assemble by
catalog membership, never by ordinal start order; it governs scale up/down only, and a
`RollingUpdate` stays strictly serial (2 × 600 s worst case).

Each compute pod also registers as a Ballista executor (`[ballista.executor]`,
`task_slots = 1`) with the scheduler's Service. `jammi-server-scheduler` is one CPU Deployment
(`[ballista.scheduler]` advertised as its Service name, no executor table): it claims a training job and places it on a
registered compute executor as one task, or runs it in-process while none is registered. A plain
Service suffices — with a single replica, a stable Service DNS name survives a restart as well as an
ordinal would, and the same name is its `peer_advertise`. With `local_ranks = 2` the overlay admits
single-pod gangs (`world_size ≤ 2`, a `Local` gang on one pod's two devices); a cross-pod `Peer`
gang of world `W` needs `max_world_size ≥ W` at the submit edge and at least `W` pods able to hold a
rank. The overlay is validated by kubeconform only — CI has no GPU node, so it never runs a pod.

### 2.2 ConfigMap key form

A ConfigMap volume materialises one file per key, named after the key. The resolution chain reads
exactly `/etc/jammi/jammi.toml`, so the key must be `jammi.toml`. A `configMapGenerator` entry
`files: [jammi.toml]` yields that key; a source file with any other name needs the
`files: [jammi.toml=<file>]` form. The base and `ci` generators name their source `jammi.toml` and
render the key `jammi.toml`.

The `shape-d` generators keep descriptive source names and rename at the key:
`files: [jammi.toml=jammi-compute.toml]` and `files: [jammi.toml=jammi-scheduler.toml]`. A bare
source name there would render a key no step of the chain reads, and the failure is quiet: the
scheduler pod would boot on `JammiConfig::default()` (all tiers, every kind claimed, no Ballista
scheduler), and the compute pod would refuse to load, because its env sets `peer_advertise` with no
`peer_bind` from a file (`crates/jammi-db/src/config/tests.rs::
load_from_peer_advertise_without_peer_bind_is_refused_naming_both_keys`). kubeconform does not look
inside `ConfigMap.data` and the kind smoke never applies `shape-d`, so the property is pinned on the
rendered manifests: `ci/scripts/check_kube_config_mounts.py` requires every ConfigMap mounted at
`/etc/jammi` to carry the file its container reads.

### 2.3 Image floor

The manifests' TOML is written for the current config schema: `[worker]`, `[lease]`,
`[distributed]`, `[gpu] devices`, `[ballista]`, and `[storage.cloud.s3]` as a nested table. The
current `JammiConfig` rejects unknown keys, so a config it does not understand is a load error
(`jammi-server: failed to load config:`, `crates/jammi-server/src/main.rs`) and a CrashLoop. An
image older than the schema fails differently. The latest release tag, `v0.49.1`, has none of those
sections and does not reject unknown keys: by its source (`crates/jammi-db/src/config.rs` at that
tag) it ignores `[worker]`, `[lease]` and the rest silently — its claim loop is governed by the older
`[training]` section — and it refuses the base and `shape-d` files at `[storage.cloud.s3]`, because
its `[storage.cloud]` is a `kind`-tagged table. (Read from the source at the tag; the image was not
run.) Measured on the registry on 2026-09-19: cu12 `:latest` is `0.49.1` (the same digest,
`sha256:f4bb7040…`), so the compute overlay's image predates the schema until the next `v*`
release; CPU `:latest` is a different digest from `0.49.1`, i.e. it was last moved by a `main`
dispatch. The remedy for a stale CPU `:latest` is a Server Image dispatch on `main`; otherwise pin
`:sha-<sha>` or an unprefixed `:X.Y.Z` at or after the first release carrying the schema. To check
an image against the floor:
`docker run --rm -v "$PWD/deploy/kubernetes/base/jammi.toml:/etc/jammi/jammi.toml:ro" <image> probe`
must not print `failed to load config` (the probe may still exit 1 for want of a server). The check
is necessary, not sufficient: an image that ignores unknown sections passes it on a file such as the
`ci` overlay's, which carries no `[storage]` block.

### 2.4 Failure modes

- Stale image: a load error and CrashLoopBackOff, or sections silently ignored (§2.3).
- Wrong ConfigMap key: the file lands at a path the chain never reads (§2.2); refused by `check_kube_config_mounts.py`.
- Missing Secret: `optional: false` leaves the pod in `CreateContainerConfigError`, never ready.
- No ordering gate: Kubernetes has no `depends_on`, so a server pod started alongside Postgres/NATS
  converges by CrashLoopBackOff-and-retry until its catalog and broker answer.
- Termination with a job in flight: DRAIN within the grace finishes the job; past the grace, SIGKILL
  sends it down the expiry path at the cost of one attempt, and the successor resumes from the last
  epoch bundle. Caps a DRAIN cannot cross: kubelet graceful node shutdown defaults to 0 s, AWS Spot
  gives two minutes' notice, GCP Spot at most 30 s — on spot capacity use RELEASE from a `preStop`
  hook (`jammi-server release` works on every image; the CPU images are distroless and have no
  shell).
- Pod replacement loses the `emptyDir`: the Hub cache is re-downloaded and, where no `[storage]`
  root is set, local result tables are gone. Persistence options are the deployer's: an RWX PVC, an
  init-container pre-pull, or `[models] hub_cache_dir` with `offline`. Compose differs: its named
  volume persists across a restart.

---

## 3. Pinned versions

| What | Pin | Where |
|---|---|---|
| `helm/kind-action` | `06c1ae10762d3b9c1644e7fe69596ae519e015a2 # v1.15.0` | `kube-smoke.yml` |
| kubectl | `kubectl_version: v1.34.11`, set explicitly — the action's default (`v1.37.0`) is three minors over the server | `kube-smoke.yml` |
| kind node image | `kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d` (from the kind v0.33.0 release notes, the kind binary `kind-action` v1.15.0 installs) | `kube-smoke.yml` |
| kubeconform | `v0.8.0`; `kubeconform-linux-amd64.tar.gz` sha256 `9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883` | `ci.yml`, `kube-smoke.yml` |
| kubeconform `-schema-location` | `https://raw.githubusercontent.com/yannh/kubernetes-json-schema/b582a12a09aa9b5d1edad577a964c504130214fc/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json` | `ci.yml`, `kube-smoke.yml` |
| `--kubernetes-version` | `1.34.11` | `ci.yml`, `kube-smoke.yml`, README |
| kustomize | `kustomize/v5.8.1`; `kustomize_v5.8.1_linux_amd64.tar.gz` sha256 `029a7f0f4e1932c52a0476cf02a0fd855c0bb85694b82c338fc648dcb53a819d` | `ci.yml`, `kube-smoke.yml` |
| Sidecar images in the `ci` overlay | `postgres:16`, `nats:2.10-alpine` (`-js -m 8222`), mirroring `deploy/docker-compose.yml` | `overlays/ci/` |

Both binaries are fetched with `curl -fsSL` and verified with `sha256sum -c` in workflow steps — no
third-party action, no distro package — so the gate's result depends on nothing that moves. The
schema cache is keyed on the schema sha plus the Kubernetes version, so a pin change starts a fresh
cache. kubeconform v0.8.0 fails hard on a missing `-cache` directory, and `actions/cache` creates it
only on a hit, so the steps `mkdir -p` it first. Node image, kubectl and `--kubernetes-version`
share one Kubernetes version so that the schema validated against is the API the smoke applies to.

**The validation invocation**, identical in `ci.yml` and `kube-smoke.yml`, for `<k>` in `base`,
`overlays/shape-d`, `overlays/ci`, under `set -euo pipefail` so a failing `kustomize build` fails
the pipe:

```
kustomize build deploy/kubernetes/<k> | kubeconform --strict --summary --kubernetes-version 1.34.11 \
  -schema-location "https://raw.githubusercontent.com/yannh/kubernetes-json-schema/b582a12a09aa9b5d1edad577a964c504130214fc/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json" \
  -cache "$RUNNER_TEMP/kubeconform-cache" -
```

Measured with kustomize v5.8.1 and kubeconform v0.8.0 on 2026-09-19: `base` 3 resources,
`overlays/shape-d` 9, `overlays/ci` 7, all valid. `replicas: "three"` fails with `at
'/spec/replicas': got string, want null or integer`; a misspelt key `replica:` fails with
`additional properties 'replica' not allowed` under `--strict` and passes without it, which is why
the gate is strict; removing `behavior: replace` from the `ci` generator fails the build with `can
not use behavior: 'unspecified', behavior must be merge or replace`. A scratch copy for such probes
must copy the whole `deploy/kubernetes` tree — an overlay copied alone cannot resolve `../../base`.

---

## 4. `kube-smoke`

The workflow mirrors `compose-smoke.yml`: resolve the CI base image to a digest once, build the CPU
image from the commit's `Dockerfile` with `push: "false"` and `load: "true"` (a literal `false`, so
the job is not a promotion primitive and needs no `packages: write`), generate an ephemeral masked
audit key, render `overlays/ci/kind-config.yaml.in` with `envsubst` (the runner's checkout is handed
to the node through `extraMounts`, and the `ci` patch mounts it as `hostPath` — the fixtures include
a directory, which a ConfigMap cannot carry), create the cluster, `kind load` the three images,
create the namespace and the Secret imperatively, validate the overlay before applying it, apply,
and wait on `rollout status` for both StatefulSets and the Deployment.

Oracles, in order:

1. **Image identity, one hop.** Every pod's `containerStatuses[].imageID` equals the built image's
   id or a `crictl` `repoDigest` on the node; an empty id list is an error, not a pass. `kind load`
   registers a tag ref only, so `repoDigests` is empty and the equality arm is the expected one.
   `imagePullPolicy: Never` is a hint, not the mechanism.
2. **The driver** (`tests/compose/shape_c_kube_remote.py`) over a `kubectl port-forward`: the
   runtime `broker == "jet_stream"`, register the `patents.parquet` fixture, embed with `tiny_bert`,
   an exact self-hit search, then `rollout restart` + `rollout status --timeout=180s`, reopen the
   forward, and the after-restart property.
3. **Catalog.** `select count(*) from sources` ≥ 1, queried in the Postgres pod — fails if the
   catalog URL were dropped from the Secret.

What it does not prove: result durability across the restart (the scratch volume is an `emptyDir`);
the value of a file-only knob (the `ci` TOML carries engine defaults; the smoke proves the file
parses and that env wins); anything about `shape-d`.

---

## 5. Smoke scripts

`tests/compose/remote_smoke.py` is the oracle: the fixture constants, `wait_for_ready`, the
end-to-end `run(target, health_url, *, restart, after_restart)` body, a `Ctx` dataclass handed to the
callback, and two named callbacks. Every assertion prints what it compared before raising.

- `durable_after_restart` (Compose, `shape_b_remote.py`): the same search answers with identical
  keys, scores within `1e-6`, and an unchanged `list_index_segments` — durability on the volume.
- `shared_catalog_after_restart` (Kubernetes, `shape_c_kube_remote.py`): `describe_source("patents")`
  is visible, the `list_sources()` count is unchanged, and the broker is still `jet_stream` — the
  new pod shares the old pod's catalog and broker. It never queries the result table: with no
  `[storage]` block the table's Parquet lives on the `emptyDir`, and
  `crates/jammi-db/src/store/mod.rs::load_existing_tables` registers only a `ready` row whose
  Parquet still exists, so asserting it would assert a property this shape deliberately lacks.

Both drivers keep a `--dry-run` that prints the plan offline. `tests/compose/test_remote_smoke.py` is
a stdlib `unittest` suite that imports no `jammi`; it runs as the `remote smoke oracle` guard in
`ci/guards.toml`.

---

## 6. Docs integration

`docs/guide/src/reference-topologies.md` includes `base/{deployment.yaml,service.yaml,jammi.toml}`
under "Kubernetes (deploy/kubernetes)" and all six `shape-d` files under "Shape D", with mdbook's
`{{#include}}`. The fence test resolves an include whose body is that single line against the
page's directory, keeps the failure location at the fence, and appends the included path to the
message. `docs.yml` watches `deploy/**` in both `paths:` lists, since a manifest edit changes the
rendered guide. `deploy/kubernetes/README.md` carries the layout, the Secret line, config precedence,
local validation, the DRAIN/RELEASE runbook with rollout arithmetic, the compute-plane roles, the
autoscaling input (`jammi_jobs_queued{kind}`, `jammi_worker_jobs_in_flight` on `/metrics`), pin
advice and the name-suffix note.

---

## 7. Properties the tests hold

- Every kustomization builds and every rendered resource is valid under the strict 1.34.11 schema —
  `Guard (kubernetes manifests)`, required by `ci-summary`.
- Every `JammiConfig`-shaped TOML the guide shows, the three included manifest files among them,
  parses under the real loader; the selected count is pinned at 30; an unknown key fails naming the
  fence — `docs_toml_fences_parse_under_the_real_loader`,
  `unknown_key_in_a_docs_shaped_fence_fails_naming_it`.
- The Compose driver passes `durable_after_restart` and issues exactly the `docker compose … restart`
  argv; the Kubernetes driver passes `shared_catalog_after_restart`, issues `rollout restart` then
  `rollout status` in that order, and never issues SQL — `ComposeDriverTests`, `KubeDriverTests`,
  `SharedCatalogAfterRestartTests::test_never_queries_sql`, `DurableAfterRestartTests`.
- The `ci` overlay on a real cluster reaches readiness on the built image, is backed by JetStream and
  Postgres at runtime, and shares its catalog across a rollout restart — `kube-smoke`.

Not pinned by any test: the rendered ConfigMap key (§2.2), and the published images against the
image floor (§2.3).

---

## 8. Outside this design

Seams the deployer owns, each named in the README or above rather than shipped: CPU/memory
resources and QoS, PodDisruptionBudget, PriorityClass, Hub-cache persistence, an autoscaling adapter
over the queue gauges, spot/preemptible pools (RELEASE via `preStop`), ingress, TLS, NetworkPolicy,
ServiceMonitor, per-cloud annotations, and the `result_root`/`region` placeholders. Not in this tree:
Helm, an operator, a second compute overlay, and any engine change — the `[worker]` defaults, the
shutdown modes and the gauges belong to the engine and are described in
`OPS-COMPUTE-TIER-OPERABILITY.md`.

## 9. References

- kubernetes.io — Downward API (`env` `fieldRef`, `$(VAR)` expansion order in `EnvVar.value`); Pod v1
  API (`env` overrides `envFrom`; `terminationGracePeriodSeconds` default 30 s); container probes
  (`failureThreshold × periodSeconds`); Deployments (selector immutable); StatefulSets (stable
  network identity, `podManagementPolicy`, `RollingUpdate` ordering); headless Services and
  `publishNotReadyAddresses`; Disruptions (a PDB is app-owner created); Pod priority and preemption
  (PriorityClass is cluster-scoped; a missing class rejects the pod); Node graceful shutdown (default
  0 s); Persistent volumes (RWX is provider-dependent); HPA (custom metrics need an adapter);
  ConfigMap volumes (key = file name). Read for Kubernetes 1.34.
- kubernetes-sigs/kustomize v5.8.1 — `configMapGenerator` `files: key=path`, `behavior: replace`,
  `generatorOptions.disableNameSuffixHash`, the `images:` transformer; `checksums.txt` for the
  release.
- kind v0.33.0 — `nodes[].extraMounts`; `kind load docker-image` registers a tag ref only; the node
  image digest in the release notes.
- yannh/kubeconform v0.8.0 — `-schema-location` template (`NormalizedKubernetesVersion` is
  `v`-prefixed), `-cache`, `-strict`; `CHECKSUMS`.
- yannh/kubernetes-json-schema@`b582a12a09aa9b5d1edad577a964c504130214fc` —
  `v1.34.11-standalone-strict/{deployment,statefulset}-apps-v1.json`.
- helm/kind-action v1.15.0 `action.yml` — inputs `node_image`, `cluster_name`, `kubectl_version`,
  `config`.
- docker/metadata-action v5.10.0 — `type=semver,pattern={{version}}` yields unprefixed tags;
  `flavor: latest=auto` emits `latest` for a non-prerelease semver tag; `{{is_default_branch}}` is
  false on a tag ref.
- cluster-autoscaler — `--max-graceful-termination-sec` default 600 s.
- huggingface_hub — `HF_HOME` / `HF_HUB_CACHE` / `HF_HUB_OFFLINE` semantics, mirrored by the
  `[models]` knobs.
