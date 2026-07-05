# Jammi Python ecosystem — the client-as-base unification

Status: **PLAN — grounded + pressure-tested.** A design + build plan for a disruptive,
two-repo refactor. It was **grounded** against the current surfaces (an API-divergence
inventory + a blast-radius/packaging inventory) and **survived an adversarial pressure-test**
(verdict REFINE; every correctness finding folded in — error-taxonomy §5.4, the 4 MB cap
§5.9, the capability contract §5.6). No code has moved.

> **Greenfield posture (per `CLAUDE.md`).** Jammi has **no production users** and takes a
> strict no-backwards-compat stance: *"No shims, no deprecated paths, no keep-the-old-way-
> around. Change it everywhere. Break and rebuild correctly. No `#[deprecated]`, no
> compatibility re-exports."* This plan assumes that posture throughout. **Migration means
> changing every call-site atomically in the same change — no shims, no aliases, no
> deprecation windows.** This is a correction from the first draft of this plan, which wrongly
> assumed backwards-compatibility (compat shims, deprecation timelines, a reversibility-gated
> Stage-1/Stage-2 split). All of that machinery is removed here: the correctness work is
> unchanged, the *phasing* is now purely about reviewable PR-sizing, and the rename is a
> product/positioning call with no compat cost — not a risk gate.

## 1. Thesis

Today the Python surface is three *parallel* packages a user perceives as different
products:
- `jammi_ai` — the **native, embedded** engine (maturin/PyO3, in-process).
- `jammi_client` — a **pure-Python remote** client to a `jammi-server`.
- `jammi_ai_platform` — the **commercial** SDK (governance), which re-exports `jammi_client`.

The insight that reorganizes all of it: **the client is the constant; the engine is a
backend that relocates.** What actually changes as a user scales is not *which package*
but *where the engine runs*:

```
embedded (in-process)  →  local server  →  remote server        (+ enterprise server → governance)
        └────────────── you write against ONE client the whole way ──────────────┘
```

This is Jammi's own law — *"one binary serves every topology via pluggable backends;
deployment is configuration, not a fork"* (`docs/guide/src/philosophy.md`) — applied one
layer up, to the SDK. The engine already obeys it; the Python surface should too.

## 2. Target state

**One client is the base.** The engine is a backend behind it. Two optional install-time
capabilities, in the two directions that are actually valid:

```
pip install jammi                 # the client — talks to any server by URL
pip install jammi[embedded]       # + the native in-process engine backend (no server)
pip install jammi-platform        # + commercial governance (plugs into the client)
```

- `jammi` — the client API you write against, always. Pure-Python. Can drive **any**
  engine location.
- `jammi[embedded]` — a **downward, OSS extra**: the base optionally pulls the heavier
  native engine wheel as an in-process (direct-FFI) backend. This is a textbook-correct
  extra (base → optional heavy dep), the *opposite* of the rejected upward `client[platform]`.
- `jammi-platform` — a **separate commercial distribution** that **plugs into** `jammi`
  via a generic hook. It is *not* an extra of `jammi` (that would name the commercial
  package in the OSS base). Once installed and pointed at an enterprise server, the
  existing `jammi` connection surfaces `registry`, `experiments`, `gates`, etc.

Deployment is a `connect()` argument:
```python
import jammi
db = jammi.connect()                       # embedded  (needs jammi[embedded])
db = jammi.connect("localhost:50051")      # local server   — same code
db = jammi.connect("grpc://prod:50051")    # remote server  — same code
# enterprise server + jammi-platform installed → db.registry / db.experiments appear
```

Analogy: the **DuckDB/SQLite-or-served, Postgres-client** model — code against the
client; the engine is embedded or served; moving between them is a connection target,
not a rewrite.

## 3. Why the two capabilities are asymmetric (and must be)

- **`[embedded]` is an extra** because it is OSS pulling OSS, pointing *down* to an
  optional heavier dependency. Correct use of extras.
- **`platform` is a plug-in package** because it is commercial and the OSS client must
  **name no consumer** (Jammi's one-way-references rule, `philosophy.md`). The client
  exposes a *generic* extension hook that names nothing; platform registers into it. So
  "don't name platform in the OSS client" holds **by construction**, not by hiding.
- **Discovery is not access.** The SDK is inert without a licensed enterprise server +
  credentials, so `jammi-platform` can be *public* on PyPI (good for top-of-funnel); the
  paywall is the license/server, not package visibility. No private index needed.

## 4. Architecture

### 4.1 The client + backend trait
`jammi` defines one API surface (`Database`/`connect` + the engine operations) over a
**backend trait** with two implementations:
- **EmbeddedBackend** — direct in-process calls into the native engine (FFI/PyO3), *not*
  gRPC-over-loopback, so embedded keeps its performance (zero-copy where possible).
- **GrpcBackend** — the wire client, for **local or remote** servers (local vs remote is
  only a URL). Enterprise is a GrpcBackend pointed at the enterprise server.

The surface must be **backend-agnostic**: one return-type story (e.g. Arrow-based) across
FFI and gRPC. Ops that are genuinely one-sided are handled explicitly (§5).

### 4.2 The platform plug-in hook
`jammi` publishes a generic extension mechanism (an entry-point group it owns, e.g.
`jammi.extensions`, and/or a `jammi.*` namespace scan) — **naming no consumer**.
`jammi-platform` registers an implementation; when present, the `jammi` connection
lights up the governance handles. Absent, the base client is unchanged. The namespace
(`jammi.platform`) is *optional polish* on top of the hook — the "add-on" experience
comes from the plug-in, not the import path.

### 4.3 Naming
- `jammi` — the client (the flagship name goes to what users import).
- `jammi[embedded]` — native in-process backend (the current `jammi-ai` native wheel,
  repositioned as a backend distribution, name TBD in the spec).
- `jammi-platform` — the commercial add-on (today's `jammi-ai-platform`).
- server — no package; a URL.

## 4a. Grounded current state (the target is mostly today's design, inverted)

The inventory changed the risk profile. The unification is **already ~90% realized** —
just packaged inside-out and enforced by duck-typing instead of a type:

- `jammi_ai.connect(target)` **already dispatches** `file://` → the in-process native
  engine and every remote scheme → `jammi_client.connect` (`python/jammi_ai/__init__.py:50`).
  `jammi_client.connect` raises `NoEmbeddedEngineError` on `file://`. So the "one connect,
  pick the backend" front door exists — it just lives in the *native* package, which
  **bundles** `jammi-client` (exact `==0.33.0` pin).
- **~40 verbs are already identical, signature-pinned** by `crates/jammi-python/tests/
  test_conformance.py`, both built on the **shared** `jammi_client._assembly.build_*_request`
  layer, converging on `pyarrow.Table` / `dict` / `str` / `List[float]`.
- The two classes (`Database`, `RemoteDatabase`) share **no base class/Protocol** — they
  agree by convention + the conformance test. **That missing shared type is the core work.**

So this refactor is **"promote a test-pinned duck-typed parallelism into a real shared
trait, and invert the packaging"** (light client base + native engine as the discovered
`[embedded]` backend) — *not* a from-scratch API merge. The `file://`-vs-remote dispatch
even points the right way already: today `jammi_client.connect(file://)` errors; the target
is that it dispatches to the embedded backend *if `jammi[embedded]` is installed*, else
errors with a `pip install jammi[embedded]` hint.

## 5. The real work (bounded, from the divergence inventory)

Not "the whole surface." A short, enumerable list:

**Trivial (the bulk — ~40 verbs):** already identical + test-pinned + shared assembly →
drop behind one trait with near-zero design.

**Moderate (bounded design):**
1. **Introduce the shared `Backend`/`Session` Protocol** both classes implement explicitly
   — removes the `__getattr__` duck-typing and the `Union[Database, RemoteDatabase]` return.
2. **Unify the divergent handle families** under handle Protocols: `TrainingJob` (PyO3) vs
   `RemoteTrainingJob` (Python); `TenantScope` (PyO3) vs the `@contextmanager` generator.
3. **`credentials` is a signature change to the single `connect` front door — lands with the
   trait (Unit 1).** Today `jammi_ai.connect` takes no `credentials` and calls
   `jammi_client.connect` positionally, so the embedded front door cannot auth its own
   remote arm. The unified `connect` grows a `credentials=` parameter; the embedded backend
   accept-and-ignores it (and rejects it on a `file://` target). This changes the front-door
   signature, so it lands with the trait (Unit 1, §6), not later.
4. **Normalize the error taxonomy (pressure-test find — a real seam, not a wrapper).**
   Embedded verbs raise `PyRuntimeError`/`PyValueError` (plus native `audit`/`ephemeral`
   sub-taxonomies); remote raises `grpc.RpcError` + `jammi_client.TrainingError`. The **same
   failure surfaces as a different exception type** per backend, and `test_conformance.py`
   pins signatures but **not** raised types — so a failed embedded training job raises a bare
   `PyRuntimeError` while remote raises `TrainingError`, and a caller's `except TrainingError`
   silently misses the embedded case. Fix: one `jammi` exception hierarchy (`JammiError` base
   → `TrainingError`, `NotSupportedOnBackend`, `InvalidArgument`→`ValueError`), map **both**
   backends onto it, and **extend `test_conformance.py` to pin the raised type per verb**, not
   just the signature.
5. **Collapse (or contractually pin) the two return-construction paths** — embed builds
   Arrow/dicts in Rust; remote hand-builds them in Python (~15 `_*_to_dict` projections).
   The *shape* is test-pinned; the two encode/decode paths are a standing drift risk.

**Decided — the capability contract (was the "capability-probing vs lying" open question):**
6. **One-sided ops are resolved by an explicit, typed capability contract, not a silent
   `AttributeError`.** The trait exposes `supports(capability)` and any one-sided call on the
   wrong backend raises a typed **`NotSupportedOnBackend`** (never a bare `AttributeError`).
   This absorbs the embedded-only capabilities (`audit` handle, `ephemeral_session`,
   `preload_model` — no remote verb; audit is remotely reachable only as a trigger *topic*)
   and the lifecycle asymmetry: `close()`/context-manager is a **no-op embedded** (RAII drop)
   / **real remote** (gRPC channel teardown); `session_id` is **real remote / `None`
   embedded**. Callers probe with `supports()` or catch `NotSupportedOnBackend` — the edge is
   honest and typed, decided here rather than left to the implementer.

**Hard (genuine reconciliation — the honest limits of "seamless"):**
7. **Streaming asymmetry:** real back-pressured subscription exists only over gRPC; embed
   is a synchronous `block_on` collect. A unified streaming iterator is new embedded design.
8. **`sql` dual transport:** same signature/return, but embed = in-process DataFusion,
   remote = **Flight SQL** (a *separate* transport with its own bearer threading). The
   unified auth/session model must span both the typed-gRPC lane and the Flight lane.
   **Caveat (external dependency):** the *server*-side BYO-auth enforcement over Flight is
   itself **unbuilt** (engine issue #220) — a client-side unification cannot close a
   server-side gap, so this item is blocked on #220 and tracked as an external dependency, not
   deliverable purely from the client.
9. **The 4 MB gRPC unary cap (pressure-test find — an honest edge, not free seamlessness).**
   Remote `infer` / `search` / `assemble_context` / `subscribe_collect` return the whole
   result as **one unary `ArrowBatch`**, bounded by gRPC's ~4 MB default receive cap; embedded
   is **unbounded**. So an identical call that succeeds embedded **fails remote** with
   `RESOURCE_EXHAUSTED` on large data — the backends are not truly interchangeable at the size
   boundary. Fix: raise `max_receive_message_length` on the `GrpcBackend` channel **and** list
   this as an explicit honest-edge of the "seamless" promise, not something the trait silently
   papers over.

**Packaging (mechanical but wide):** invert so the light pure-Python client is the base;
reposition the native `jammi_ai` wheel as the `[embedded]` backend (new native module
namespace, abi3, Linux+macOS wheels); rework `jammi-ai-platform` from "re-exports
`jammi_client`" to "plugs into `jammi`"; migrate the call-sites (**48 `jammi_ai` + 32
`jammi_client` + 27 `jammi_ai_platform`**) and ~700 doc lines. Every one of these sites is
**in-repo**, so each rename rewrites all of its call-sites in the *same* change — the old
names are deleted, not aliased.

**The cookbook is a co-evolution surface, not a find-replace.** The counts above are
accurate — ~46 files under `cookbook/` touch the `jammi_ai`/`jammi_client` surface — but
framing them as low-risk call-sites is *wrong*. Under the engine↔cookbook loop — *every new
engine surface earns a measured cookbook chapter; the cookbook is the engine's validator* —
sed-ing `jammi_ai`→`jammi` ships the rename but **fails to prove the new surface**, which
violates the loop's law. This unification is not a rename dressed as a migration: it
introduces genuinely new surface — the single `jammi` client, `connect()` as
backend-relocation, the `supports()`/`NotSupportedOnBackend` capability contract (§5.6), the
`JammiError` taxonomy (§5.4), the `credentials=` front door (§5.3), and the honest edges (the
4 MB unary cap §5.9, the streaming asymmetry §5.7) — and **each earns a measured chapter**.
The cookbook carries its own mechanized harness (`cookbook/book/scripts/check_api_reference.py`,
`cookbook/book/tests/test_rails.py`, `test_closed_loop.py`, `test_channels.py`); those chapters
are proven by that harness running green against the *new* surface. So the cookbook is a
first-class per-unit deliverable and a merge gate (§6), not incidental blast radius.

## 6. Build units (clean greenfield, atomic rewrites — no shims)

There are no production users and every call-site is in-repo, so there is **no compat path
to preserve** — a big-bang is safe. The only reason to split the work into units at all is
**manageability**: each unit is a self-contained, reviewable, correct PR-sized change that
runs the full rigor chain and leaves the tree working (all call-sites of anything it touches
migrated, old names *deleted*). The unit count is a PR-sizing call, **not** a compat
requirement — the whole thing could land in one PR if that were reviewable.

Because the cookbook is the engine's validator (§5), **every unit's rigor chain gates on the
cookbook loop closing** — `test_rails` / `test_closed_loop` / `check_api_reference` green
against the *new* surface, not merely `cargo test` + `test_conformance.py` green. The cookbook
co-evolves *inside* the unit that changes the surface it teaches; a unit is not done when the
rename compiles, but when its measured chapters prove the new surface.

A lean, sensible split is **three units**:

- **Unit 1 — unified `jammi` base + shared trait + the §5 correctness work.** Establish the
  `jammi` client and the shared `Backend`/`Session` trait as the canonical home by **moving**
  `RemoteDatabase` + the shared `_assembly` layer **into** `jammi` and **deleting the old
  locations** (`jammi_client` is renamed into `jammi`, not aliased from it — every
  `jammi_client` import site is rewritten in this same change). This unit lands the trait, the
  normalized error hierarchy (§5.4) with raised-type conformance, the
  `supports()`/`NotSupportedOnBackend` capability contract (§5.6), and the `credentials=`
  front-door signature change (§5.3). Because there is no downward alias, there is no import
  cycle to design around — the code simply lives in `jammi` and the old module names cease to
  exist. Ships with the GrpcBackend. **Cookbook co-evolution:** this unit rewrites
  `quickstart/01_install.md` + `02_connect.md` as the flagship client-as-base narrative (the
  single `jammi` front door, `connect()` as backend-relocation) and adds measured chapters for
  the capability contract (show `supports()` and catch `NotSupportedOnBackend` when an
  embedded-only op is called on a remote handle) and the remote honest edge — per the
  no-silent-caps rule, SHOW the 4 MB unary call returning `RESOURCE_EXHAUSTED` over the
  GrpcBackend (§5.9). The unit merges only when these chapters and the cookbook harness are
  green against the new surface.
- **Unit 2 — the `[embedded]` backend under the new native namespace.** Add the
  EmbeddedBackend (direct-FFI over the native engine) behind the same trait and ship
  `jammi[embedded]`. The native extension module lives at the top-level `jammi_native`
  namespace (maturin `module-name`), moved off the old dotted submodule path which is now
  **gone** — every direct native-import site (e.g. `test_conformance.py`'s `_NativeDatabase`)
  imports it as a bare top-level module. `jammi_ai` is not kept alongside; it is replaced. **Cookbook co-evolution:** the
  chapters that teach the embedded path — installing the `[embedded]` extra, the direct-FFI
  backend, and the embedded↔remote parity the trait now guarantees — co-evolve here, measured
  green against the real embedded backend this unit ships. This unit also completes the §5.9
  honest-edge chapter by showing the same 4 MB call *succeeding* embedded, closing the contrast
  against the remote `RESOURCE_EXHAUSTED` half landed in Unit 1.
- **Unit 3 — `jammi-platform` plug-in + the rename/PyPI claim + docs.** Add the generic
  extension hook to `jammi`; rework `jammi-ai-platform` → `jammi-platform` to register into
  it (a rename + rework across the enterprise repo, **not** an alias — the 27
  `jammi_ai_platform` sites are rewritten). This unit also carries the fresh `jammi` PyPI
  claim and the ~700 doc-line migration. **Cookbook co-evolution:** the governance chapters —
  the plug-in lighting up `registry` / `experiments` / `gates` on a `jammi` connection, named
  as generic primitives with no consumer — co-evolve here, and this unit gates on *those*
  governance chapters green. The rename is a product/positioning decision (§8),
  so this unit can be sequenced whenever convenient — even first — since it carries no compat
  cost that would force an ordering.

Call-site counts to migrate (the tests are a mechanical rewrite; the cookbook share is *not*
a find-replace — it co-evolves as a measured per-unit deliverable, §5 and each unit above),
all in-repo and all rewritten atomically within the unit that owns them:
**48 `jammi_ai`** (2 library, ~35 cookbook, 11 tests), **32 `jammi_client`** (4 library —
the local↔remote seam in `python/jammi_ai/`, the rest tests/cookbook), **27
`jammi_ai_platform`** (4 library, 23 tests, all enterprise). Note: `jammi-enterprise`
imports **zero `jammi_ai`** (it goes through `jammi_client`), so the engine rename touches
enterprise only via the `jammi-client` distribution dependency, not its code.

## 7. Cross-repo sequencing, releases, PyPI names (grounded)

Current distributions / versions / triggers:

| PyPI name | Import | Ver | Repo | Kind | Publish trigger |
|---|---|---|---|---|---|
| `jammi-ai` | `jammi_ai` | 0.33.0 | jammi-ai | native abi3 wheels (Linux, macOS arm64/x86_64; Win disabled) | tag `py-v*` (`pypi.yml`) |
| `jammi-client` | `jammi_client` | 0.33.0 | jammi-ai `clients/python` | pure-Python | **same `py-v*` tag** (`pypi-client.yml`) |
| `jammi-server`(+`-cu12`) | `jammi_server` | 0.33.0 | jammi-ai `packaging/` | binary-carrying wheel | `pypi-server*.yml` |
| `jammi-ai-platform` | `jammi_ai_platform` | 0.7.0 | jammi-enterprise | pure-Python (proprietary) | tag `sdk-v*` (`sdk-pypi.yml`) |

Load-bearing coordination facts:
- **`py-v*` publishes `jammi-ai` AND `jammi-client` together** (two workflows, one tag),
  and `jammi-ai` pins `jammi-client==0.33.0` exactly. A rename must **retag + rewire both
  atomically**.
- **`jammi-ai-platform` versions independently** (`0.7.0`, separate `sdk-v*`, separate
  repo) and depends on `jammi-client>=0.21` (a **floor**, not the exact pin). A
  `jammi-client` rename breaks the SDK's dependency string + its 4 library import sites —
  the cross-repo coordination point.
- **`jammi` is a fresh name** — not used anywhere in the project (no dist, no module, no
  npm). A top-level `jammi` PyPI claim is external: check availability on pypi.org and
  register a **pending publisher** (all four workflows use OIDC Trusted Publishing under a
  `pypi` environment — a new name needs the pending-publisher set up before first publish).
- The native module name `jammi_native` is set in `[tool.maturin] module-name`; a
  package rename touches maturin config + `python/jammi_ai/__init__.py`, not just the dist.

**Release order:** the OSS base (`jammi` + `jammi[embedded]`, engine repo, `py-v*` line)
leads; `jammi-platform` (enterprise repo, `sdk-v*` line) follows the unit it depends on. The
rename retires the old `jammi-ai`/`jammi-client`/`jammi-ai-platform` distributions outright —
there is no parallel-publishing window, because there are no users pinned to the old names to
carry.

## 8. The substantive work vs. the rename (a product call, not a risk gate)

Two distinct things live in this plan: the **unification value** (a shared trait, normalized
errors, the capability contract, the credentials fix, the embedded backend, the platform
plug-in) and the **rename** (`jammi_ai`→`jammi` + the fresh `jammi` PyPI claim). The first
draft gated the rename behind a reversibility "kill-switch" (stop at Stage 1, keep aliases so
everything reverts). **Under greenfield posture that framing is void:** there are no users to
disrupt and nothing to revert *to* (no shims, no aliases keeping the old names alive), so the
rename carries no risk cost to gate against.

- **The unification is the work.** The shared `Backend`/`Session` trait, the normalized error
  hierarchy (§5.4), the `supports()`/`NotSupportedOnBackend` capability contract (§5.6), the
  `credentials=` front-door fix (§5.3), the embedded backend, and the platform plug-in hook
  are the substance. Each ships behind its own audit + green CI.
- **The rename is a product/positioning decision, standalone on its own merits.** Adopting
  the flagship `jammi` import name and claiming the fresh `jammi` PyPI name buys
  **positioning** — a single obvious name users write against. It adds no capability the
  unification didn't already ship. Because it carries **no compat cost** (every call-site is
  in-repo and rewritten atomically), it is not risk-gated: it can be sequenced whenever
  convenient — bundled with the unification, done first, or done last — purely on whether the
  positioning is worth the coordination (the cross-repo `sdk-v*` dance, the PyPI claim, the
  ~700 doc lines). The only real precondition is external: that the `jammi` PyPI name is
  actually available (§7, §9).

**Cross-repo caveat (confirm before the rename):** the enterprise-SDK facts the rename
depends on — the `jammi-client>=0.21` floor and the 27 `jammi_ai_platform` sites — live in
the **other repo** and are **not verifiable from this tree**. They are load-bearing
assumptions to re-confirm against the enterprise repo before executing the rename, not
established facts.

## 9. Genuinely open questions

The pressure-test resolved most of what this section used to ask. *Is "seamless" achievable?*
— **yes, with an honest, typed edge**: ~40 verbs converge on one return story and are
signature-pinned; the one-sided capabilities are enumerable and now handled by the **decided**
capability contract (§5.6). *Capability-probing vs. lying?* — **decided**: `supports()` +
typed `NotSupportedOnBackend`, never a silent `AttributeError`. *Rename now vs. later?* —
**decided** (§8): under greenfield posture the rename is a standalone product/positioning call
with no compat cost, so it can be sequenced whenever convenient — it is not a risk gate. The
divergence-completeness worry is addressed by the new §5 items (error taxonomy, 4 MB cap).

What genuinely remains open:

- **`jammi` PyPI availability** (external check) — if the top-level `jammi` name is
  unavailable on pypi.org, the whole naming premise needs a fallback. Cannot be settled from
  this tree.
- **Does the `jammi.platform` namespace earn its restructure,** or does the generic plug-in
  hook alone suffice (likely the latter, per §4.2)? Open naming/structure call, low stakes.
- **The enterprise-SDK cross-repo facts** — the `jammi-client>=0.21` floor and the 27
  `jammi_ai_platform` sites — must be **confirmed against the enterprise repo** before the
  rename (§8); they are not verifiable from this tree.
