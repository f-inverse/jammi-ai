# CONTRACT — feat/500-C-U5b-1a: gang-membership substrate

**Contract of record.** slug: `feat_500-C-U5b-1a` · branch `feat/500-C-U5b-1a`
at `d0606edeee01d2972023855eccd3d53aebf385ba` · this file is the committed
mechanism contract `ci/scripts/check_rigor_record.py` requires under
`docs/rigor/contracts/**` (its check 3) before this unit's rigor record at
`docs/rigor/feat_500-C-U5b-1a.jsonl` (the lead's export) satisfies check 1/2.
Source design contract: the lead's `CONTRACT-U5b-1a.md` v4 (scratchpad-only,
never a repo path — it is not cited with a `path:line` token anywhere in this
document, since it is not a tracked file at HEAD and any such token naming it
would fail this checker's own cost-floor pass).

**Citation convention (round-2 addendum, binding for this revision).** A
whole FUNCTION or TYPE is cited BY CONSTRUCT —
`` `path/to/file.rs::Type::method` `` or `` `path/to/file.rs::Type` `` —
never by a line range, so an unrelated edit inserting lines earlier in the
same file (A5's `GangCandidateRow` struct is the exact class of churn that
forced this convention: it shifted every later `jobs_repo.rs` function's
line number by +12/13 and the round-1 contract kept the pre-shift numbers in
§1.3 while §4 had already been corrected) can never leave a stale citation
behind. `path:line`/`path:a-b` is reserved for STATEMENT-level claims: a
specific SQL string, a specific conjunct/`continue` arm, a specific
`sort_by` call, a specific error-message text, a specific doc-comment
sentence being quoted. Every `path:line`/`path:a-b` token in this revision
was re-derived directly against the tree of the code commit named above (not
carried from the round-1 text, not carried from the source design contract's
own, differently-based, line numbers) and self-checked — see §6.

This document states the mechanism as it EXISTS at the commit named above
(CURRENT-STATE, per CLAUDE.md — no "this was added because..." framing
beyond what a citation needs to be unambiguous).

## Scope

U5b-1a lands the gang-membership substrate DESIGN.md § 4 and DIST § 5.8
sketch: two new nullable columns on `instances` (`peer_addr`, `result_root`),
the ONE registration carrier every writer of the row builds
(`InstanceRegistration`) and its ONE choke point (`from_config`), the two read
verbs (`peer_addr_of`, `list_gang_members`), the lease keeper's whole-tuple
reregister on a missed heartbeat, and the config-side canonicalization
(`[server] peer_advertise`, `JammiConfig::canonical_result_root`). It does
**not** build a caller of either read verb (that is U5b-1b-ii's coordinator)
or the attestation VERIFY that gives canonical-root equality sufficiency
(U5a-1's whole-artifact sidecar / U5b-0's and U5b-1b-i's per-partition
inventory already exist as separate mechanisms; this unit only states that
root equality alone is never sufficient).

---

## 1. Mechanism

### 1.1 The carrier: `PeerAddr`, `CanonicalRoot`, `WorkerFacts`, `InstanceRegistration`

`crates/jammi-db/src/catalog/instance.rs`, a module.

- **`PeerAddr`** (`crates/jammi-db/src/catalog/instance.rs::PeerAddr`) is a
  sealed wrapper (`pub struct PeerAddr(String)`, private field): the only way
  to build one is `PeerAddr::parse`
  (`crates/jammi-db/src/catalog/instance.rs::PeerAddr::parse`), which
  `rsplit_once(':')`s the input so an IPv6 literal's own colons stay inside
  the host segment and only the LAST colon splits host from port, requires a
  non-empty host, and a port that parses as `u16` and is non-zero. `as_str()`
  and `Display` both return the validated wire form. This is the SAME type
  `crates/jammi-db/src/index/peer.rs` re-exports (per this module's own doc
  comment) — the peer listener and the gang listener dial the identical
  address type, so they can never drift into two.
- **`CanonicalRoot`** (`crates/jammi-db/src/catalog/instance.rs::CanonicalRoot`)
  wraps an already-canonicalized root string; `Self::new` exists for
  `JammiConfig::canonical_result_root`/`MembershipConfig::materialize` (§1.2,
  §1.4) and fixtures, not as a general-purpose constructor any caller may
  build an unvalidated string through.
- **`WorkerFacts`** (`crates/jammi-db/src/catalog/instance.rs::WorkerFacts`)
  is the claim-loop half of a registration: `kinds: String` (the comma-joined
  kind set, discussed at §2 below) and `state: WorkerState`. Its doc states
  its SOLE owner is `JobWorker`/`EmbeddedWorker`
  (`crates/jammi-ai/src/fine_tune/worker.rs`), confirmed at §1.5.
- **`InstanceRegistration`**
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration`):
  `instance_id`, `label`, `host`, `peer_addr: Option<PeerAddr>`,
  `canonical_root: Option<CanonicalRoot>`, and `worker:
  Mutex<Option<WorkerFacts>>` — the claim-loop cell, mutated in place by its
  single owner rather than requiring a whole new registration per state
  change. `InstanceRegistration::new`
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::new`) is
  the plain constructor with no validation, used by `from_config` itself and
  by fixtures/tests (confirmed in `crates/jammi-db/tests/it/gang_membership.rs`'s
  `seed_member` helper and direct-construction call sites, and
  `crates/jammi-ai/tests/it/instance_identity.rs`'s foreign-row fixture).
  `set_worker`/`worker_snapshot`
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::set_worker`,
  `::worker_snapshot`) are the cell's read/write pair (mutex-poisoning
  tolerant via `unwrap_or_else(|p| p.into_inner())`).
- **`GangListing<'a>`** (`crates/jammi-db/src/catalog/instance.rs::GangListing`)
  and **`GangMember`** (`crates/jammi-db/src/catalog/instance.rs::GangMember`)
  are `list_gang_members`'s request/response shapes (§1.3).

### 1.2 The PURE/MATERIALIZE split: `MembershipConfig::validate` and `InstanceRegistration::from_config`

**F1 fix.** Before this split, `JammiConfig::load_from` called the SAME
`InstanceRegistration::from_config` the session did, and `from_config`'s
anchor check REQUIRED the anchor to already exist on disk — so on a
genuinely fresh host (`artifact_dir` not yet created) `load_from` refused,
while `InferenceSession::new`/`open` on the IDENTICAL config silently
accepted it, because `JammiSession::new`'s own catalog open
(`crates/jammi-db/src/session.rs::Catalog::open_with_tenant`'s
`std::fs::create_dir_all(artifact_dir)`) had already created the anchor as a
side effect BEFORE `wrap_with`'s `from_config` call ever ran — two
enforcement points, two different verdicts on one config. The fix splits the
check into a PURE validate half and a MATERIALIZE half.

**§9 redesign (round-2 BLOCK on M5, this revision).** The round-1 fix bound
`MembershipConfig::validate`'s absoluteness/UTF-8 guards to
`StorageUrl::parse(artifact_dir.to_string_lossy())` — a URL-reinterpreted
VIEW of `artifact_dir` — while every other consumer of it (the catalog's own
directory creation, `resolved_result_root`, the local cache dir, `JobWorker`)
reads the literal `PathBuf`. Round-2's audit found, with executed probes:
(F-A1) `artifact_dir = "file:///var/lib/jammi"` passed `is_absolute()` as a
URL while the literal `PathBuf` every OTHER consumer reads is RELATIVE;
(F-A2) a cloud-spelled `artifact_dir` took the `Cloud` arm, which cannot
carry the leaf, so the canonical string named `s3://b/p` while the store
actually roots at `s3://b/p/jammi_db` — a false-positive gang member;
(F-A3) the lossy fold sat on the COMPARED value. The redesign, in
`crates/jammi-db/src/catalog/instance.rs::MembershipConfig::validate`
(PURE — no filesystem read/write of any kind):

```
peer_advertise unset  => Ok(None)                                   // non-member
peer_advertise = addr => peer_bind must be Some, else JammiError::Config
                         naming both keys (instance.rs:304-308)
                      => PeerAddr::parse(addr)?                     (:309)
                      => [storage] result_root UNSET: the anchor IS
                         artifact_dir's LITERAL PathBuf — checked with
                         Path::is_absolute() (:315-321) and
                         Path::to_str().is_some() (:322-327), NEVER
                         reinterpreted as a URL
                      => [storage] result_root SET: parsed VERBATIM via
                         StorageUrl::parse (:335-339, no scheme
                         lowercasing); memory:// refused (:341-347);
                         file:// requires the WHOLE path absolute
                         (:350-356); a cloud scheme keeps StorageUrl's
                         own case-preserved rendering (:359-362)
                      => Ok(Some(MembershipConfig { peer_addr, effective_root }))
```

`EffectiveRoot` (`crates/jammi-db/src/catalog/instance.rs::EffectiveRoot`)
has three variants matching the three arms above: `Default(PathBuf)` (the
literal `artifact_dir`), `ResultRootFile(PathBuf)` (the whole absolute path
an explicit `file://`/bare-path `result_root` names — no leaf, since
`result_root` already names the whole effective root), and
`ResultRootCloud(String)` (`StorageUrl`'s own verbatim, case-preserved
rendering — no leaf, trailing `/` trimmed; an uppercase scheme token is
refused by `StorageUrl::parse` itself, case-sensitively, exactly as
`build_result_store` would refuse the SAME string — never silently folded).

`JammiConfig::load_from`
(`crates/jammi-db/src/config/mod.rs::JammiConfig::load_from`) calls ONLY
`MembershipConfig::validate`, discarding the `Option`
(`crates/jammi-db/src/config/mod.rs:2600`) — loading a config file must
never itself create a directory as a side effect, so a bad
`peer_advertise`/`peer_bind` pair or a RELATIVE anchor is refused at load,
naming the key, while a merely-MISSING (but absolute, well-formed) anchor is
ACCEPTED — the pure check has nothing to say about existence at all.

`InstanceRegistration::from_config`
(`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::from_config`)
= `MembershipConfig::validate` PLUS `MembershipConfig::materialize`
(`crates/jammi-db/src/catalog/instance.rs::MembershipConfig::materialize`):
for a `file://` anchor (either arm), the shared helper
`materialize_file_anchor`
(`crates/jammi-db/src/catalog/instance.rs::materialize_file_anchor`) checks
the anchor is not already a non-directory FIRST (so the precise "must be a
directory" refusal beats `create_dir_all`'s own less legible OS error text),
`std::fs::create_dir_all`s it if absent (idempotent with `JammiSession`'s own
`create_dir_all` of `artifact_dir`, and with `ResultStore`'s later one of the
same `result_root` path — refusing naming the key if it cannot be created),
then `std::fs::canonicalize`s it and UTF-8-checks the result via
`into_os_string().into_string()` — never `to_string_lossy()`. The `Default`
arm alone appends the `jammi_db` leaf, lexically, AFTER canonicalizing
`artifact_dir` (`crates/jammi-db/src/catalog/instance.rs:394-401`) — never
itself resolved, whether absent, present, or a symlink. This is the ONLY
constructor `Catalog::upsert_instance`/`Catalog::reregister_instance` accept
(enforced by their own signatures, §1.3 — both take `&InstanceRegistration`,
never raw fields).

**Partial-tree honesty.** `MembershipConfig::materialize`'s own doc states
`create_dir_all` is not atomic: a failure partway through (a permission
fault, a concurrent removal) can leave some parent directories created on
disk even though the call returns `Err` — stated as fact, never papered
over.

`crates/jammi-ai/src/session.rs::InferenceSession::wrap_with` calls
`from_config` (`crates/jammi-ai/src/session.rs:232-237`), BEFORE the lease
keeper starts (`crates/jammi-ai/src/session.rs::LeaseKeeper::start`'s call
site), before the result store creates any directory (`build_result_store`'s
call site), and before any prune/reclaim sweep — but `wrap_with` receives an
ALREADY-CONSTRUCTED `JammiSession` (`inner`, its parameter), whose catalog
connection is already open and whose `artifact_dir` the caller's own
`JammiSession::new` already created; a `from_config` refusal here returns
`Err` without closing that connection, so "runs before wrap_with's own side
effects" is not the same claim as "leaves nothing behind" (§2.8b A2 in
`docs/maintainer/MAINTAINER-GUIDE.md` restates this precisely;
`crates/jammi-ai/tests/it/instance_identity.rs::file_result_root_anchor_fails_open_and_writes_no_row`
reopens the SAME catalog directory in a bounded retry loop after a
deliberately-failed construction for exactly this reason). `wrap_with` is
the universal funnel: every `InferenceSession` constructor (`new`, `open`,
`open_with_placement`) reaches it, confirmed by
`crates/jammi-ai/src/session.rs::InferenceSession::open`,
`::open_with_placement`, and the private `::wrap`, all of which call
`wrap_with` either directly or through `wrap`.

`ServerConfig::validate`
(`crates/jammi-db/src/config/mod.rs::ServerConfig::validate`, the 3-way
listener-collision check) is NOT the home for either half: it has no access
to `artifact_dir`/`storage.result_root`, which the anchor check needs.

**P-B2 purity, restated.** Given an ABSOLUTE anchor (guaranteed by F2's
check), `MembershipConfig::materialize`'s output depends on nothing but the
config:
`crates/jammi-db/src/config/tests.rs::canonical_result_root_is_independent_of_a_chdir_between_load_and_materialize`
proves it survives a `std::env::set_current_dir` between `validate` and
`materialize` — guarded by a local `Mutex` since cwd is process-global.

### 1.3 The two read verbs and the two writers

`crates/jammi-db/src/catalog/jobs_repo.rs`:

- **`upsert_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::upsert_instance`): one
  `INSERT ... VALUES (...) ON CONFLICT(instance_id) DO UPDATE SET label,
  host, peer_addr, result_root, last_seen_at`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2093-2099`) — `started_at` is
  stamped only on the initial insert (never in the `DO UPDATE SET` list).
  `reg.peer_addr`/`reg.canonical_root` write `NULL` for a non-member
  registration.
- **`reregister_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::reregister_instance`):
  the SAME `instances` upsert as `upsert_instance`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2138-2144`), plus, INSIDE THE
  SAME TRANSACTION, an `INSERT ... ON CONFLICT(instance_id) DO UPDATE SET
  kinds, state` into `workers` — but ONLY when `reg.worker_snapshot()` (taken
  once, before the transaction) is `Some`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2155-2167`, the `if let
  Some(w) = worker { ... }` guard). This is the ONLY verb that re-creates a
  pruned row; `touch_instance`
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::touch_instance`, a
  pure `UPDATE ... WHERE instance_id = $2`) can never resurrect one
  (`Ok(updated == 1)`, `false` when the row is absent).
- **`fresh_instance(instance_id, lease)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fresh_instance`,
  unchanged from base except its doc): `true` iff a row is present AND NOT
  `stale_before_clause("last_seen_at", kind, margin, ...)`, where `margin =
  instance_liveness_margin(lease)` (`super::lease`, §1.6). No tenant
  predicate — `instances` carries no tenant column.
- **`peer_addr_of(instance_id, lease) -> Result<Option<PeerAddr>>`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::peer_addr_of`):
  `SELECT peer_addr FROM instances WHERE instance_id = $N AND peer_addr IS
  NOT NULL AND NOT ({stale})`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2269-2272`); `None` when the row
  is absent, stale, or `peer_addr` is NULL, ALL collapsed the same way (a
  single `query_opt`, no distinguishing return). A stored `peer_addr` that
  fails `PeerAddr::parse` is a typed `JammiError::Catalog`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2281-2285`), never silently
  mapped to `None` — a corrupted row fact is distinct from an absent/stale
  one. No kind/root/self filter: this is the ONE by-id resolution surface,
  used by a rank resolving its own coordinator or any member resolving any
  other by id, including a busy or other-kind one.
- **`list_gang_members(GangListing)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::list_gang_members`):
  SQL (`crates/jammi-db/src/catalog/jobs_repo.rs:2325-2332`) is `SELECT
  i.instance_id, i.peer_addr, i.result_root, w.kinds, w.state FROM instances
  i JOIN workers w ON w.instance_id = i.instance_id WHERE i.peer_addr IS NOT
  NULL AND i.result_root IS NOT NULL AND NOT ({stale})` — an INNER join (a
  member must have a `workers` row: a fleet worker with a claim-loop slot),
  and freshness/NULL-ness pushed into SQL while everything else stays in
  Rust. The Rust filter loop, over the named `GangCandidateRow` struct (A5 —
  no tuple positions to confuse), excludes, in order: the caller itself
  (`instance_id == listing.self_instance`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2351-2353`); `state !=
  WorkerState::Claiming.as_db_str()`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2354-2356`, so
  `warming`/`draining` are excluded); `kinds.split(',').map(str::trim).any(|t|
  t == listing.kind)` false
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2357-2364`, a WHOLE trimmed
  token, never a substring — `fine_tune` cannot match `graph_fine_tune`);
  `result_root.as_bytes() != listing.canonical_root.as_str().as_bytes()`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2365-2367`, a Rust byte
  comparison, never SQL `=`). A surviving row's `peer_addr` is parsed
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2368-2373`, the same
  typed-error-on-corruption shape as `peer_addr_of`) and pushed as a
  `GangMember`. The final list is `members.sort_by(|a, b|
  a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()))`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2379`) — Rust byte-order sort,
  never a SQL `ORDER BY` (backend collation is untrusted).
- **`upsert_worker`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::upsert_worker`),
  **`set_worker_state`** (`::set_worker_state`), and **`delete_worker`**
  (`::delete_worker`) are unchanged in SQL shape from base; their callers
  moved (§1.5). `upsert_worker`'s own doc and `WorkerFacts::kinds`'s doc both
  still read "the comma-joined (or otherwise producer-encoded) kind set"
  verbatim — this phrase is NOT tightened by this unit (the verb itself
  performs no separator validation); what IS pinned is that the sole
  production producer, `JobWorker::run_until`/`EmbeddedWorker::begin_drain`
  (`crates/jammi-ai/src/fine_tune/worker.rs:822,827,851`), always calls
  `self.kinds.join(",")`, so `list_gang_members`'s comma-split-whole-token
  match is exercised against a comma-joined string in every real deployment,
  never merely asserted possible.
- **`prune_instances(stale_after)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::prune_instances`):
  `DELETE FROM instances WHERE {stale}` — its own `workers` row cascades
  (`ON DELETE CASCADE`, confirmed at `crates/jammi-db/src/catalog/schema.rs:1030`,
  `workers.instance_id`'s FK, declared in the same migration 029 block).

### 1.4 `[server] peer_advertise` and `JammiConfig::canonical_result_root`

`crates/jammi-db/src/config/mod.rs`:

- **`ServerConfig::peer_advertise: Option<String>`**
  (`crates/jammi-db/src/config/mod.rs::ServerConfig::peer_advertise`, field
  at `crates/jammi-db/src/config/mod.rs:1636`, doc `:1618-1635`): "the
  address OTHER replicas dial THIS process's `peer_bind` listener at ...
  `None` (the default) = this process never advertises a gang membership
  row: its `instances.peer_addr`/`result_root` columns stay `NULL`
  regardless of whether `peer_bind` is set. Requires `peer_bind` to be set
  too (refused at the ONE membership choke point,
  `InstanceRegistration::from_config` — naming BOTH keys)".
- **`resolved_result_root(&self) -> Result<String>`**
  (`crates/jammi-db/src/config/mod.rs::JammiConfig::resolved_result_root`):
  `storage.result_root` when `Some`, else `self.artifact_dir.join("jammi_db")`'s
  UTF-8 string — the SAME derivation `ResultStore::new`'s local-root arm
  performs. A1 fix: fallible — refuses naming `artifact_dir` when the joined
  path is not valid UTF-8
  (`crates/jammi-db/src/config/mod.rs:2483-2488`), never
  `Path::to_string_lossy`'s silent replacement-character fold (which could
  make two genuinely different paths compare equal downstream).
- **`canonical_result_root(&self) -> Result<Option<CanonicalRoot>>`**
  (`crates/jammi-db/src/config/mod.rs::JammiConfig::canonical_result_root`,
  `crates/jammi-db/src/config/mod.rs:2521-2526`): a thin convenience wrapper
  — `MembershipConfig::validate(self)?`, then, if `Some`, `.materialize()?`
  — over the split §1.2 states in full; `Ok(None)` when `peer_advertise` is
  unset. `resolved_result_root`'s two arms are still the effective root this
  canonicalizes: `{artifact_dir}/jammi_db` (the `artifact_dir` anchor —
  `MembershipConfig`'s LITERAL `PathBuf`, never a URL reparse — created if
  absent, canonicalized, leaf appended lexically) when `result_root` is
  unset; `result_root` VERBATIM, parsed through `StorageUrl::parse` with NO
  scheme lowercasing (§9), when set: `file://` names `result_root` VERBATIM
  (the SAME string `ResultStore::with_root` roots the store at — no suffix);
  a cloud scheme is `StorageUrl`'s own case-preserved rendering,
  trailing-`/`-trimmed, with no local filesystem step at all.
- **`JammiConfig::load_from`**'s `MembershipConfig::validate` call is at
  `crates/jammi-db/src/config/mod.rs:2600` (§1.2) — NOT `from_config`, per
  the F1 fix: the PURE half only.

### 1.5 The claim-loop's ownership of the worker half

`crates/jammi-ai/src/fine_tune/worker.rs`:

- **`JobWorker::run_until`**
  (`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_until`): sets
  the registration's worker cell to `Some(WorkerFacts { kinds:
  self.kinds.join(","), state: WorkerState::Warming })`
  (`crates/jammi-ai/src/fine_tune/worker.rs:822-823`) BEFORE issuing the
  first `upsert_worker`
  (`crates/jammi-ai/src/fine_tune/worker.rs:827`) — so a keeper reregister
  racing this very first upsert can never observe an empty cell. After the
  worker gate opens, the SAME cell-before-row order flips the cell to
  `Claiming` (`crates/jammi-ai/src/fine_tune/worker.rs:851`) before calling
  `set_worker_state`.
- **`EmbeddedWorker`**
  (`crates/jammi-ai/src/fine_tune/worker.rs::EmbeddedWorker`) holds
  `registration: Arc<InstanceRegistration>`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2687`), cloned from
  `session.instance_registration()` at spawn
  (`crates/jammi-ai/src/fine_tune/worker.rs:2738`) — the SAME `Arc` the
  session's own `LeaseTarget::Instance` hold renews (confirmed:
  `crates/jammi-ai/src/session.rs:371` holds
  `LeaseTarget::Instance(Arc::clone(&registration))` from the SAME
  `registration` variable stored as `instance_registration: registration` at
  `crates/jammi-ai/src/session.rs:390`). `begin_drain`
  (`crates/jammi-ai/src/fine_tune/worker.rs::EmbeddedWorker::begin_drain`)
  preserves the cell's `kinds`, flips only `state` to `Draining`, THEN calls
  `set_worker_state` — cell before row, same ordering. `stop_and_join`
  (`crates/jammi-ai/src/fine_tune/worker.rs::EmbeddedWorker::stop_and_join`)
  clears the cell (`self.registration.set_worker(None)`,
  `crates/jammi-ai/src/fine_tune/worker.rs:2850`) BEFORE
  `self.catalog.delete_worker(&self.instance_id)`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2851`) — cell before delete,
  once the loop task has fully returned, so nothing can race a re-set of the
  cell after the clear.
- **`InferenceSession::instance_registration()`**
  (`crates/jammi-ai/src/session.rs::InferenceSession::instance_registration`)
  is the `pub(crate)` getter both `EmbeddedWorker::spawn` and the lease
  keeper's hold construction (`crates/jammi-ai/src/session.rs:371`) read
  from — the SAME `Arc` throughout a session's lifetime.

### 1.6 The prune window and the keeper's reregister

`crates/jammi-db/src/catalog/lease.rs`:

- **`instance_liveness_margin(lease) -> Duration`**
  (`crates/jammi-db/src/catalog/lease.rs::instance_liveness_margin`):
  `lease.saturating_mul(2)` — unchanged from base (this unit only adds a
  second production caller, `fresh_instance`/`peer_addr_of`/`list_gang_members`,
  to the one U5a-1 already created).
- **`instance_prune_window(lease) -> Duration`**
  (`crates/jammi-db/src/catalog/lease.rs::instance_prune_window`), NEW:
  `instance_liveness_margin(lease).saturating_add(lease)`, i.e. `3 × lease` —
  STRICTLY beyond the margin (the self-test
  `crates/jammi-db/src/catalog/lease.rs::tests::prune_window_is_strictly_beyond_the_liveness_margin`,
  which asserts `window == 3×lease` exactly and `window > margin` for `{1,
  5, 30, 3600}` second leases, plus the `Duration::MAX` saturating-add case).
  The doc states the defect this closes explicitly: before this function
  existed, the only caller (`InferenceSession::wrap_with`) pruned at exactly
  the liveness margin, so a merely-stale member was ALREADY prune-eligible
  the instant it read stale.
- **`InferenceSession::wrap_with`**'s prune call
  (`crates/jammi-ai/src/session.rs:342-344`):
  `catalog.prune_instances(jammi_db::catalog::lease::instance_prune_window(
  lease_intervals.lease())).await?` — the named function, never a literal
  `saturating_mul(2)`/`(3)` at the call site.
- **`renew_all`'s `Instance` arm**
  (`crates/jammi-db/src/catalog/lease_keeper.rs::renew_all`):
  `touch_instance(&reg.instance_id)`
  (`crates/jammi-db/src/catalog/lease_keeper.rs:772`) → `Ok(true)` renews;
  `Ok(false)` (a missed touch — the row was pruned during a transient
  outage) calls `catalog.reregister_instance(reg)`
  (`crates/jammi-db/src/catalog/lease_keeper.rs:780`) INSTEAD of flipping
  `lost` — `Ok(())` is treated as renewed, an `Err` warns and neither
  renews nor flips `lost` (same shape as a faulted `touch_instance`). The
  `Some(true)`/`Some(false)`/`None` dispatch
  (`crates/jammi-db/src/catalog/lease_keeper.rs:851-855`) confirms: `None`
  is a genuine no-op, never a silent `lost` flip on a keeper-side fault.
- **`LeaseTarget::Instance(Arc<InstanceRegistration>)`**
  (`crates/jammi-db/src/catalog/lease_keeper.rs::LeaseTarget::Instance`,
  `crates/jammi-db/src/catalog/lease_keeper.rs:67`) carries the `Arc`, not
  an owned copy — the SAME registration `JobWorker` mutates in place is the
  one this hold renews, so a state change lands on the very next renewal
  with no re-registration.

---

## 2. Properties (quantified, never a single-input claim)

**P-M3 (the membership predicate, restated over the shipped shape).** For
every `list_gang_members(GangListing { kind, self_instance, canonical_root,
lease })` call and every DB row order: a row is returned iff it is NOT
`self_instance`, has a `workers` row (INNER join) with `state == Claiming`,
`kinds` contains `kind` as a whole trimmed comma-split token (a kind that is
merely a SUBSTRING of a token, e.g. `fine_tune` vs. `graph_fine_tune`, is
excluded — `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_kind_that_is_only_a_substring_token`),
`peer_addr` and `result_root` are both non-NULL, `result_root` bytes equal
`canonical_root` bytes (a root divergent only by case or a trailing `/` is
excluded —
`crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_root_divergent_by_case_or_trailing_slash`),
and it is fresh under `instance_liveness_margin(lease)`; the surviving set
is sorted by `instance_id` byte order regardless of the underlying row order
(proven against a DESCENDING-insertion-order fixture with a raw-SELECT
vacuity control,
`crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order`).
`peer_addr_of` resolves a busy or other-kind fresh member (no kind/state
filter) and is `None` for a stale or NULL-`peer_addr` one
(`crates/jammi-db/tests/it/gang_membership.rs::peer_addr_of_resolves_a_busy_or_other_kind_fresh_member`
and the three `peer_addr_of_is_none_for_*` siblings). Both verbs return the
SAME answer under a scoped tenant binding and under none
(`crates/jammi-db/tests/it/gang_membership.rs::list_gang_members_is_identical_under_a_scoped_tenant_and_under_none`,
mirroring `Catalog::get_job_for_rank`'s own tenant-independence oracle).

**P-M4 (recovery restores the WHOLE membership tuple).** For a process that
IS a gang member (a non-NULL `peer_addr`/`result_root` row AND a `workers`
row with `state == Claiming`), force-deleting its `instances` row and running
one real lease-keeper pass makes `list_gang_members` return it again, with
`peer_addr`, `result_root`, `kinds`, and `state` byte-identical to before
the delete — proven twice, at the catalog layer with a hand-built keeper
(`crates/jammi-db/tests/it/gang_membership.rs::keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete`)
and at the session layer through the session's OWN keeper, no hand-built one
(`crates/jammi-ai/tests/it/instance_identity.rs::a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass`).
A `fresh_instance`-only assertion is NOT this oracle — this unit's oracles
assert the `workers` row's `kinds`/`state` too, not merely the `instances`
row's presence. The dual is proven too: a DRAINED worker (its registration
cell cleared, its `workers` row deleted) is NOT resurrected as a member after
a forced delete of its `instances` row — the keeper reregisters the process
(it is still alive) but never re-inserts a `workers` row, so the INNER join
still excludes it
(`crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`).

**P-prune-window.** A member merely stale in `(instance_liveness_margin(lease),
instance_prune_window(lease)]` is NOT pruned; one strictly beyond the window
IS — proven at the catalog layer directly
(`crates/jammi-db/tests/it/gang_membership.rs::prune_window_does_not_prune_a_member_merely_stale_within_the_window`)
and through the REAL construction sweep, where only the RIGHT function
(`instance_prune_window`, `3×lease`) — never the old literal
`saturating_mul(2)` (which equals the margin itself) — can leave such a row
standing
(`crates/jammi-ai/tests/it/instance_identity.rs::a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep`).

**P-M5 (over the constructor, restated for §9).** `peer_advertise` without
`peer_bind` → typed error naming both keys
(`crates/jammi-db/src/config/tests.rs::from_config_peer_advertise_without_peer_bind_is_refused_naming_both_keys`,
and at `load_from` directly,
`crates/jammi-db/src/config/tests.rs::load_from_peer_advertise_without_peer_bind_is_refused_naming_both_keys`,
`crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_without_peer_bind_fails_open_naming_both_keys`,
through the REAL `InferenceSession::new`); a RELATIVE `artifact_dir` or
explicit `result_root` → typed error naming the key, PURE (no filesystem
access)
(`crates/jammi-db/src/config/tests.rs::membership_config_validate_refuses_a_relative_artifact_dir`,
`::membership_config_validate_refuses_a_relative_result_root`, and at
`load_from` directly,
`::load_from_refuses_a_relative_artifact_dir_when_peer_advertise_is_set`); a
`file://`- or cloud-scheme-SPELLED `artifact_dir` → ALSO refused as
relative, naming `artifact_dir` — never silently reinterpreted as a URL
(§9's F-A1/F-A2 closure —
`crates/jammi-db/src/config/tests.rs::membership_config_validate_refuses_a_file_url_spelled_artifact_dir_as_relative`,
`::membership_config_validate_refuses_a_cloud_url_spelled_artifact_dir_as_relative`);
a non-UTF-8 `artifact_dir` → refused naming the key, never a lossy fold on
the compared value (F-A3 —
`crates/jammi-db/src/config/tests.rs::membership_config_validate_refuses_a_non_utf8_artifact_dir`,
built via `OsString::from_vec` on unix); a MISSING (but absolute,
well-formed) anchor → CREATED, never refused — F1's own fix
(`crates/jammi-db/src/config/tests.rs::from_config_creates_a_missing_artifact_dir_anchor`,
`::canonical_result_root_creates_a_missing_explicit_result_root_anchor`, and
at `load_from` directly,
`::load_from_accepts_a_fresh_missing_anchor_when_peer_advertise_is_set`,
which additionally asserts the directory is NOT created — the pure half
never touches the filesystem —
`crates/jammi-ai/tests/it/instance_identity.rs::missing_result_root_anchor_is_created_and_session_open_succeeds`,
through the REAL `InferenceSession::new`, which DOES materialize it); the
anchor's PARENT being a plain FILE (ENOTDIR, a structural failure distinct
from a permission fault, so it holds under root CI) → refused naming the
key
(`crates/jammi-db/src/config/tests.rs::materialize_refuses_when_the_anchor_cannot_be_created_parent_is_a_file`);
a non-directory (FILE) anchor itself → refused naming the key, for both
`artifact_dir` and an explicit `result_root`
(`crates/jammi-db/src/config/tests.rs::from_config_anchor_that_is_a_file_is_refused_naming_the_key`,
`::canonical_result_root_refuses_a_file_anchor_for_an_explicit_result_root`,
`crates/jammi-ai/tests/it/instance_identity.rs::file_result_root_anchor_fails_open_and_writes_no_row`,
which additionally reopens the SAME catalog directory after the failed
construction and asserts ZERO `instances` rows — the failed check runs
before any write); `result_root` UNSET → accepted, canonicalizing
`{artifact_dir}/jammi_db`
(`crates/jammi-db/src/config/tests.rs::from_config_unset_result_root_canonicalizes_artifact_dir_jammi_db`,
`crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_set_result_root_unset_produces_a_nonnull_row_via_open`,
through the REAL `InferenceSession::open`); `result_root` SET to a `file://`/
bare path (an existing directory) → accepted, canonicalizing THAT root
VERBATIM, no `jammi_db` leaf
(`crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_set_result_root_set_produces_a_nonnull_row_via_open_with_placement`,
through `InferenceSession::open_with_placement`); `result_root` SET to a
cloud scheme → parsed VERBATIM, no scheme lowercasing, an uppercase scheme
refused consistently with the store's own parse of the identical string
(§9 —
`crates/jammi-db/src/config/tests.rs::canonical_result_root_parses_a_cloud_result_root_verbatim_no_scheme_lowercasing`,
`::canonical_result_root_refuses_an_uppercase_cloud_scheme_consistently_with_the_store`);
a library config (no `peer_advertise`) → NULLs, never an error
(`crates/jammi-db/src/config/tests.rs::from_config_without_peer_advertise_is_a_library_registration_with_nulls`,
`crates/jammi-ai/tests/it/instance_identity.rs::library_config_without_peer_advertise_writes_null_membership_columns`,
through a real session). **F1's own oracle** — the two enforcement points
reach the SAME verdict on one config —
(`crates/jammi-ai/tests/it/instance_identity.rs::load_from_and_session_open_agree_on_a_fresh_artifact_dir`)
drives the REAL public `JammiConfig::load_from` (over a real TOML file on a
fresh `artifact_dir`) AND `InferenceSession::new` on the SAME config,
asserting `load_from` creates nothing and the session writes a non-NULL
member row.

**P-B2 (the canonical root is a pure function of the config, GIVEN the
anchor; §9 redesign).** `canonical_result_root()` returns the identical
string whether the default `jammi_db` leaf is absent, present, or a symlink
to elsewhere
(`crates/jammi-db/src/config/tests.rs::canonical_result_root_is_identical_whether_the_leaf_is_absent_present_or_a_symlink`);
two spellings of one anchor (`./`, a trailing `/`, a doubled `/`) fold to the
identical string
(`crates/jammi-db/src/config/tests.rs::canonical_result_root_folds_dot_slash_and_trailing_slash_anchor_spellings`);
the canonical string equals `canon ∘ resolved_result_root()` for EVERY arm —
local default, `file://` explicit, and (new in §9) cloud explicit, in one
test
(`crates/jammi-db/src/config/tests.rs::canonical_result_root_equals_the_canonicalized_effective_root_for_every_arm`);
`memory://` is refused for a gang member
(`crates/jammi-db/src/config/tests.rs::canonical_result_root_refuses_a_memory_scheme`).
The property SURVIVES a `chdir` between load and materialize, given the
absolute anchor F2 guarantees
(`crates/jammi-db/src/config/tests.rs::canonical_result_root_is_independent_of_a_chdir_between_load_and_materialize`
— guards its own `std::env::set_current_dir` window with a local `Mutex`
since cwd is process-global).

---

## 3. Oracles, by name, and what each EXCLUDES

Every oracle below is cited BY CONSTRUCT (`path::test_fn_name`), per this
revision's citation convention (see the header) — never by a line range, so
an unrelated line-shift elsewhere in the same file can never leave the
table stale.

| Oracle | Asserts | Excludes |
|---|---|---|
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_the_caller_itself` | The self row is never returned even though it is otherwise a valid member | Does not test a self row that is ALSO stale/wrong-kind (isolates the self-exclusion conjunct alone) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_stale_member` | A row past the liveness margin is excluded | Does not probe the exact margin boundary — that is `gang_instance_freshness.rs`'s job |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_draining_worker` / `::list_excludes_a_warming_worker` | `state != Claiming` excludes, for BOTH non-claiming states independently | Neither combines a non-claiming state with a second false conjunct |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_kind_that_is_only_a_substring_token` | `fine_tune` never matches a `graph_fine_tune` token | Does not test a kind that is a PREFIX/SUFFIX in the other direction (the split-and-exact-match shape makes the direction irrelevant, but only one direction is fixtured) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_root_divergent_by_case_or_trailing_slash` | A byte-divergent root (case, trailing `/`) is excluded — proving the comparison is byte-exact, never a normalized/case-insensitive one | Does not test a root divergent by a symlink or a `..`-equivalent path (that is `canonical_result_root`'s own canonicalization job, upstream of this comparison) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_null_peer_addr` | `peer_addr IS NULL` excludes even with a fresh, correctly-kinded, claiming, root-matching row otherwise | Does not test `result_root IS NULL` in the same row (next oracle isolates it) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_member_with_peer_addr_set_but_result_root_null` | The asymmetric NULL case (migration 035 has no paired CHECK, so this row IS representable) is still excluded | Does not test the reverse (`result_root` set, `peer_addr` NULL) — the SQL predicate is symmetric (`AND` of both `IS NOT NULL`), so this direction stands for both |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_an_instance_with_no_workers_row` | The INNER join excludes an `instances` row with no `workers` row at all (not merely a non-claiming one) | Does not test a `workers` row for a DIFFERENT instance_id that happens to collide on kind (join key is exact) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_includes_a_fresh_multi_kind_claiming_worker` | The positive control: every conjunct held DOES return the member — proving the exclusions above are not vacuously satisfied by a fixture that could never pass anyway | Does not vary which of the two-or-more kinds is queried (asserts inclusion for one of them) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order` | Byte-order sort holds even when the raw SELECT (forced off a covering-index scan by selecting a second, non-indexed column) returns descending-insertion order — with an explicit control that the RAW order is asserted NOT already ascending, so the sort assertion is not vacuous | Does not test the postgres planner's own default order (both backends run this fixture; the control makes the backend's actual order irrelevant to the claim) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_gang_members_is_identical_under_a_scoped_tenant_and_under_none` | Tenant-independence, mirroring `get_job_for_rank`'s own oracle | Does not test under `with_admin_scope` (no admin-scope predicate exists on this path to begin with, unlike the gang RPC's) |
| `crates/jammi-db/tests/it/gang_membership.rs::peer_addr_of_resolves_a_busy_or_other_kind_fresh_member` | No kind/state filter on this verb — a draining, other-kind row still resolves by id | Does not test a `NULL` `peer_addr` row (next oracle) |
| `crates/jammi-db/tests/it/gang_membership.rs::peer_addr_of_is_none_for_a_stale_instance` / `::_for_a_null_peer_addr` / `::_for_an_absent_instance` | Three independent `None`-producing causes, each isolated | Does not distinguish the three by return shape — that IS the property (all `None`) |
| `crates/jammi-db/tests/it/gang_membership.rs::keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete` | P-M4 at the catalog layer: `peer_addr` byte-identical via `peer_addr_of` too (not only via `list_gang_members`), after a force-delete and one real keeper pass | Does not exercise the session-construction path (next file does) |
| `crates/jammi-db/tests/it/gang_membership.rs::prune_window_does_not_prune_a_member_merely_stale_within_the_window` | Both boundary directions: `(margin, window]` survives, past-window is pruned — a symmetric before/after assertion in one test | Does not test EXACTLY at the window boundary (only strictly inside vs. strictly past) |
| `crates/jammi-ai/tests/it/instance_identity.rs::library_config_without_peer_advertise_writes_null_membership_columns` | A real session with no `peer_advertise` writes NULL/NULL | Does not test a session that LATER sets `peer_advertise` (no such runtime mutation path exists — config is fixed at session construction) |
| `crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_set_result_root_unset_produces_a_nonnull_row_via_open` | Arm (a) through `InferenceSession::open`: the row's `result_root` equals `canonical_result_root()`'s own value | Does not test `open_with_placement` (next oracle) |
| `crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_set_result_root_set_produces_a_nonnull_row_via_open_with_placement` | Arm (b) through `open_with_placement`, and explicitly asserts NO `jammi_db` leaf in the written value | Does not test a cloud-scheme `result_root` through this real-construction path (config/tests.rs covers cloud schemes at the pure-function layer only) |
| `crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_without_peer_bind_fails_open_naming_both_keys` | The typed-error path through `InferenceSession::new`, not merely the pure `MembershipConfig::validate` unit test | Does not test the SAME failure through `open`/`open_with_placement` (all three funnel through the same `wrap_with`, so one is representative) |
| `crates/jammi-ai/tests/it/instance_identity.rs::missing_result_root_anchor_is_created_and_session_open_succeeds` | F1's fix: a missing (but absolute, well-formed) anchor is CREATED, never refused — session open succeeds and the row is non-NULL | Does not test a non-directory (vs. missing) anchor through this real-construction path (`file_result_root_anchor_fails_open_and_writes_no_row`, below, covers that arm) |
| `crates/jammi-ai/tests/it/instance_identity.rs::load_from_and_session_open_agree_on_a_fresh_artifact_dir` | F1's own oracle, stated directly: `JammiConfig::load_from` on a fresh anchor succeeds AND creates nothing, and `InferenceSession::new` on the SAME config materializes it and writes the non-NULL row — the two enforcement points agree | Does not drive `load_from`'s file-resolution search path (`JAMMI_CONFIG`, `./jammi.toml`, …) — an explicit path is passed, matching every other test in this file |
| `crates/jammi-ai/tests/it/instance_identity.rs::file_result_root_anchor_fails_open_and_writes_no_row` | The ONE case a missing/creatable anchor can never be confused with: an existing FILE anchor is refused naming the key, and the row is NEVER written — proven by reopening the SAME catalog directory and counting `instances` rows | Does not test the same anchor shape for `artifact_dir` (unset `result_root`) through this real-construction path (config/tests.rs covers both anchors at the pure-function layer) |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass` | P-M4 through the session's OWN keeper (never a hand-built one), with a real `EmbeddedWorker` | Does not test a session with NO worker spawned (the drained-worker oracle, next, covers the no-`workers`-row-after-recovery case) |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete` | The `instances` row DOES reregister (the process is alive) but `workers` does NOT re-appear — the INNER join's exclusion holds even after a real recovery | Does not test a re-spawn of a NEW worker after the drain (a fresh `upsert_worker` would naturally re-include it; this oracle is about the drained state persisting through ONE recovery, not about a subsequent claim-loop restart) |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep` | The construction-sweep prune call uses `instance_prune_window`, not the old `saturating_mul(2)` literal, through a REAL second session's boot — A3 fix: a longer (9 s) lease and an `ago` biased a quarter of the `(margin, window]` gap past `margin`, widening the real-wall-clock slack before the boot sweep runs from ~1.5 s to ~6.75 s so it is never red under load | Does not test the symmetric past-window-is-pruned case through this real path (the catalog-layer oracle above covers both directions; this one isolates the regression the function extraction fixes) |
| `crates/jammi-db/tests/it/migrations.rs::migration_035_is_ordered_after_034_and_adds_instances_peer_addr_result_root` | Ordering (relative position, never `.last()`); both columns nullable on both dialects; a both-NULL row and a both-set row are both valid inserts (no paired CHECK, unlike migration 034) | Does not test a row with exactly one of the two set through THIS oracle — `list_excludes_a_member_with_peer_addr_set_but_result_root_null` (gang_membership.rs) covers that shape's LISTING behavior, not the schema's acceptance of it (which this oracle's own both-null/both-set inserts imply by omission: no CHECK exists to reject the mixed case, and no test asserts the mixed insert fails, which would be the wrong assertion) |
| `crates/jammi-db/tests/it/migrations.rs::migration_029_copies_training_jobs_rows_into_jobs_as_queued` | The fourth K5 pin site gets teeth: `035` is on the ledger's manufactured-pre-029 DELETE list, so the reopened, replayed `instances` table carries both new columns | Proves the pin site's ABSENCE would be RED (an omission leaves the columns missing) — does not itself remove the entry to observe the red; that was done once at authorship time per the source contract's own methodology precedent, not re-executed by this file |
| `crates/jammi-db/src/catalog/lease.rs::tests::instance_liveness_margin_is_twice_the_lease` / `::prune_window_is_strictly_beyond_the_liveness_margin` | The `2×`/`3×` factors and the `Duration::MAX` saturating-arithmetic edge for both functions | Does not test a lease of zero (a lease is validated positive upstream, `LeaseConfig`'s own validation, not re-proven here) |
| `crates/jammi-db/src/catalog/instance.rs::tests::registration_worker_half_starts_unpopulated_and_is_settable` | The cell's own set/clear/snapshot round-trip in isolation | Does not test concurrent set/snapshot from two threads (the `Mutex` makes this a liveness, not a correctness, concern — untested here) |
| `crates/jammi-db/src/catalog/instance.rs::tests::peer_addr_parses_host_port` / `::_keeps_ipv6_host_intact` / `::_refuses_no_colon` / `::_refuses_empty_host` / `::_refuses_zero_port` / `::_refuses_non_numeric_port` / `::_refuses_out_of_range_port` | Every `PeerAddr::parse` edge, including the IPv6-colon-preservation shape (`rsplit_once`) | Does not test a hostname requiring DNS resolution (this type never resolves — a wire-form validator only, per its own doc) |
| `crates/jammi-db/src/config/tests.rs::materialize_refuses_when_the_anchor_cannot_be_created_parent_is_a_file` | The ENOTDIR oracle (§9 advisory): a path COMPONENT is a plain file, so `create_dir_all` structurally cannot create anything under it — refused naming the key, and this holds under root CI (a permission fault would be bypassed by root; ENOTDIR never is) | Does not test a permission-denied ("cannot be created") case separately — that arm is not independently oracled under this unit's hermetic lane, since CI's own root permission-fault caveat (this repo's own `chmod_bypassed` convention) would make it unreliable there |

---

## 4. Mutations executed / implied by the oracle shapes above

- **The self-exclusion / state-exclusion / kind-token-match / root-byte-compare
  conjuncts.** Each has its own isolated oracle (§3 table) holding every
  other conjunct at its satisfied value and flipping exactly one — deleting
  any single `continue` arm in `list_gang_members`'s Rust filter loop
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2351-2367`, over the named
  `GangCandidateRow` A5 introduced — no tuple positions to confuse) flips
  exactly the corresponding test from a correct exclusion to a false
  inclusion, never a combined probe that could hide which conjunct actually
  excluded.
- **The `sort_by` call.** Deleting
  `crates/jammi-db/src/catalog/jobs_repo.rs:2379` (or replacing it with a SQL
  `ORDER BY`) is exactly what
  `crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order`'s
  raw-order control is built to catch: the control proves the RAW select is
  not already sorted, so if the Rust sort were removed the final assertion
  (ascending byte order) would fail loudly on the very fixture engineered to
  defeat a covering-index scan.
- **`instance_prune_window` vs. the old literal.** Reverting
  `crates/jammi-ai/src/session.rs:342-344` to
  `lease_intervals.lease().saturating_mul(2)` (the pre-unit shape) is exactly
  what
  `crates/jammi-ai/tests/it/instance_identity.rs::a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep`
  is built to catch — the fixture's row is seeded stale at
  `margin + (window - margin) / 4` (A3: a QUARTER of the gap past `margin`,
  not the midpoint — widening real-wall-clock slack before `window`) so the
  old literal (equal to the margin) would prune it and the new function (the
  window) would not.
- **`reregister_instance`'s worker-half conditional.** Removing the `if let
  Some(w) = worker { ... }` guard at
  `crates/jammi-db/src/catalog/jobs_repo.rs:2155-2166` (always writing a
  `workers` row regardless of the cell) would flip
  `crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`'s
  expected "no workers row after recovery" to a false resurrection — the
  drained state (cell cleared) is the fixture that makes this conjunct
  observable.
- **The choke-point ordering.** Moving `from_config`'s call in `wrap_with`
  (`crates/jammi-ai/src/session.rs:232-237`) to AFTER `build_result_store`
  would let a FILE-anchor refusal occur only after `build_result_store` had
  ALREADY materialized the wrong thing at that path (a `jammi_db` file
  where a directory belongs, corrupting a later reader) rather than being
  caught first, on the config AS GIVEN — this fix's own F1 oracle no longer
  distinguishes "missing" from "ordering" (a missing anchor is accepted
  either way, since `MembershipConfig::validate` never checks existence and
  `from_config`'s materialize `create_dir_all`s it regardless of ordering);
  the ordering claim now rests on the FILE-anchor case alone —
  `crates/jammi-ai/tests/it/instance_identity.rs::file_result_root_anchor_fails_open_and_writes_no_row`
  is built on the CURRENT ordering and does not itself re-probe the
  ordering directly (it asserts the OUTCOME: zero rows after failure), so
  this is a documented, not executed, mutation.
- **The literal-`PathBuf` absoluteness check vs. a URL reparse (§9).**
  Reverting `MembershipConfig::validate`'s default arm
  (`crates/jammi-db/src/catalog/instance.rs:311-329`) to parse
  `artifact_dir` through `StorageUrl::parse` (the round-1 shape) is exactly
  what
  `crates/jammi-db/src/config/tests.rs::membership_config_validate_refuses_a_file_url_spelled_artifact_dir_as_relative`
  and
  `::membership_config_validate_refuses_a_cloud_url_spelled_artifact_dir_as_relative`
  are built to catch: both fixtures pass `is_absolute()` under a URL
  reinterpretation while failing it as the literal `PathBuf`, so reverting
  the check would flip both from a correct refusal to a false acceptance
  (F-A1/F-A2, restated as an executed mutation rather than only a probed
  finding).

---

## 5. What is NOT in this unit

- **Any caller of `peer_addr_of`/`list_gang_members`** — both are
  unreachable from any public RPC at this head, vacuously (no RPC calls
  either); U5b-1b-ii owes the server-side unreachability oracle when it adds
  the first caller (§2, stated non-guarantee, restated at
  `docs/guide/src/security.md`'s new I-GANG bullet, this unit's docs commit).
- **The attestation VERIFY** that gives canonical-root equality sufficiency
  — U5a-1's whole-artifact sidecar and U5b-0's/U5b-1b-i's per-partition
  inventory are separate, already-existing or separately-planned mechanisms;
  this unit states the necessary-never-sufficient boundary but builds no new
  verify path.
- **`RendezvousPlacement`** and any query-time consumption of the ring
  (DIST's own unit, downstream of this substrate).
- **A capability-scoped ring** — a replica that sets `peer_advertise` to be
  gang-reachable also joins the retrieval ring by construction (both read
  from the same `instances` columns); scoping the two independently is a
  later plan's follow-on, not this unit's.
- **Round-3 stop rule (pre-committed, source contract §9).** A THIRD BLOCK
  on M5 excises canonicalization entirely — the member row would carry the
  VERBATIM `resolved_result_root()` string and the predicate would become
  byte-equality on it (still necessary-never-sufficient) — filed as its own
  unit; no fourth round on M5.

---

## 6. Report

`impossibility_claims`:
- "no caller other than `InstanceRegistration::from_config` builds a value
  `Catalog::upsert_instance`/`Catalog::reregister_instance` accept" — not
  independently enumerated by a code-scanning oracle in this unit (unlike
  U5a-1's `get_job_for_rank` oracle); the claim rests on the two verbs' own
  signatures (`&InstanceRegistration`, a type with no public field access to
  `peer_addr`/`canonical_root` outside the crate — both are private-ish by
  convention, constructible only via `InstanceRegistration::new` or
  `::from_config`) rather than an executed grep-shaped enumeration —
  `uncovered` by this unit's own test suite; a future unit adding a second
  production call site of either verb would not be mechanically caught the
  way U5a-1's `get_job_for_rank` oracle catches a second caller of that verb.
- "a stored `peer_addr` that fails to parse is never silently treated as
  absent" — executed attempt: both `peer_addr_of`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2281-2285`) and
  `list_gang_members` (`crates/jammi-db/src/catalog/jobs_repo.rs:2368-2373`)
  return a typed `JammiError::Catalog` on a parse failure rather than
  mapping it to `None`/exclusion — verified by direct read of both call
  sites; no test manufactures a corrupted `peer_addr` row to observe the
  error (`uncovered`: the typed-error path itself is read-verified, not
  test-exercised, in this unit's own suite).
- "the `,` kind-encoding convention is honored by every real writer" —
  executed check: `crates/jammi-ai/src/fine_tune/worker.rs:822,827,851` are
  the ONLY three call sites of `self.kinds.join(",")` feeding
  `upsert_worker`/`WorkerFacts.kinds` in this crate (confirmed by direct
  read of the file, not by a code-scanning oracle — `uncovered` as a
  standing tripwire the way U5a-1's enumerating-caller oracle is one).
- (round-2 addendum) "the literal `artifact_dir` `PathBuf` and the URL a
  `StorageUrl` reparse would produce agree on absoluteness for every real
  deployment" — this claim is FALSE, and §9's redesign exists precisely
  because it is false (F-A1/F-A2): `artifact_dir` is therefore NEVER parsed
  as a URL at all, closing the claim rather than covering it.

`citations_reanchored`: every citation in this revision was re-derived
directly against the tree of `d0606edeee01d2972023855eccd3d53aebf385ba` (the
code commit named in the header) in this worktree by the writing agent — not
carried forward from the round-1 text, the source design contract's own
line numbers, or any other prior state of this file.

**Self-check (round-2 addendum requirement).** Before this revision was
committed, every `path:line`/`path:a-b` token in this file was extracted and
its cited line(s) printed against the tree of the commit named in the
header, and each was eyeballed against the identifier/claim the surrounding
sentence names:

```
python3 /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/12f161bf-7977-4318-a34d-d2eec24a0620/scratchpad/logs/db-selfcheck-citations.py
```

(a scratch script, never committed — it is not part of this repository's own
tooling; it walks the contract file for backtick-quoted `path:N`/`path:N-M`
tokens, including bare `:N` continuations of the most recently named path,
and prints each cited line verbatim from the working tree). Every token the
script found resolved to the exact statement it was cited for (a SQL string,
a specific conjunct/`continue` arm, a specific `sort_by`/error-message/
doc-comment line, a struct-field line) — none pointed at unrelated code. No
`path:line` token remained for a whole-function/type reference; every such
reference in this revision is a construct citation instead (see the header's
citation convention).

---

## 7. Gates

`cargo fmt --all --check`; `cargo clippy -p jammi-db -p jammi-ai -p
jammi-server --all-targets -- -D warnings`; `cargo test -p jammi-db -p
jammi-ai` and `cargo test --workspace` (a server-startup-adjacent config
change — other crates' test harnesses spawn the binary with their own env);
the live-postgres lane (`--features live-postgres-tests`, `JAMMI_TEST_PG_URL`
set) for every `test_case`-parameterized test in
`crates/jammi-db/tests/it/gang_membership.rs` and
`crates/jammi-db/tests/it/migrations.rs`'s 035 oracle;
`python3 ci/scripts/check_doc_parity.py`; `python3 ci/scripts/perf/
check_citations.py`; `python3 ci/scripts/check_no_consumer_names.py`;
`python3 ci/scripts/check_swarm_bijection.py`; `RUSTDOCFLAGS="-D warnings"
cargo doc --no-deps -p jammi-db -p jammi-ai`.
