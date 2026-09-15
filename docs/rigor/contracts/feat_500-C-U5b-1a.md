# CONTRACT — feat/500-C-U5b-1a: gang-membership substrate

**Contract of record.** slug: `feat_500-C-U5b-1a` · branch `feat/500-C-U5b-1a`
at `21a4bd0c3f193b9bc7b873db1176e4d6760952ba` · this file is the committed
mechanism contract `ci/scripts/check_rigor_record.py` requires under
`docs/rigor/contracts/**` (its check 3) before this unit's rigor record at
`docs/rigor/feat_500-C-U5b-1a.jsonl` (the lead's export) satisfies check 1/2.
Source design contract: the lead's `CONTRACT-U5b-1a.md` v3 (scratchpad-only,
never a repo path — it is not cited with a `path:line` token anywhere in this
document, since it is not a tracked file at HEAD and any such token naming it
would fail this checker's own cost-floor pass). Every citation below is a full
repo-relative path, never a bare filename, tagged **(at 21a4bd0c)**, and was
read directly against that commit in this worktree by the agent writing this
file — not carried forward from the source contract's own (differently-based)
line numbers.

This document states the mechanism as it EXISTS at `21a4bd0c` (CURRENT-STATE,
per CLAUDE.md — no "this was added because..." framing beyond what a citation
needs to be unambiguous).

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

`crates/jammi-db/src/catalog/instance.rs` (at 21a4bd0c), a new module.

- **`PeerAddr`** (`crates/jammi-db/src/catalog/instance.rs:36`) is a sealed
  wrapper (`pub struct PeerAddr(String)`, private field): the only way to
  build one is `PeerAddr::parse` (`crates/jammi-db/src/catalog/instance.rs:44-64`),
  which `rsplit_once(':')`s the input so an IPv6 literal's own colons stay
  inside the host segment and only the LAST colon splits host from port
  (`crates/jammi-db/src/catalog/instance.rs:45`), requires a non-empty host
  (`:48-52`) and a port that parses as `u16` and is non-zero (`:53-62`).
  `as_str()` (`:67-69`) and `Display` (`:72-76`) both return the validated
  wire form. This is the SAME type `crates/jammi-db/src/index/peer.rs`
  re-exports (per this module's own doc comment,
  `crates/jammi-db/src/catalog/instance.rs:1-20`) — the peer listener and the
  gang listener dial the identical address type, so they can never drift into
  two.
- **`CanonicalRoot`** (`crates/jammi-db/src/catalog/instance.rs:91-103`) wraps
  an already-canonicalized root string; `Self::new` (`:95-97`) exists for
  `JammiConfig::canonical_result_root`/`MembershipConfig::materialize` (§1.2,
  §1.4) and fixtures, not as a general-purpose constructor any caller may
  build an unvalidated string through.
- **`WorkerFacts`** (`crates/jammi-db/src/catalog/instance.rs:120-126`) is the
  claim-loop half of a registration: `kinds: String` (the comma-joined kind
  set — see the doc comment at `:121-122`, discussed at §2 below) and
  `state: WorkerState`. Its doc (`:111-118`) states its SOLE owner is
  `JobWorker`/`EmbeddedWorker` (`crates/jammi-ai/src/fine_tune/worker.rs`),
  confirmed at §1.5.
- **`InstanceRegistration`** (`crates/jammi-db/src/catalog/instance.rs:141-151`):
  `instance_id`, `label`, `host`, `peer_addr: Option<PeerAddr>`,
  `canonical_root: Option<CanonicalRoot>`, and `worker:
  Mutex<Option<WorkerFacts>>` — the claim-loop cell, mutated in place by its
  single owner rather than requiring a whole new registration per state
  change (doc, `:147-149`). `InstanceRegistration::new`
  (`crates/jammi-db/src/catalog/instance.rs:160-175`) is the plain
  constructor with no validation, used by `from_config` itself and by
  fixtures/tests (confirmed in `crates/jammi-db/tests/it/gang_membership.rs:89-95`
  and `crates/jammi-ai/tests/it/instance_identity.rs`'s foreign-row fixture).
  `set_worker`/`worker_snapshot` (`crates/jammi-db/src/catalog/instance.rs:179-189`)
  are the cell's read/write pair (mutex-poisoning tolerant via
  `unwrap_or_else(|p| p.into_inner())`).
- **`GangListing<'a>`** (`crates/jammi-db/src/catalog/instance.rs:416-430`)
  and **`GangMember`** (`:435-438`) are `list_gang_members`'s request/response
  shapes (§1.3).

### 1.2 The PURE/MATERIALIZE split: `MembershipConfig::validate` and `InstanceRegistration::from_config`

**F1 fix (this commit).** Before this split, `JammiConfig::load_from` called
the SAME `InstanceRegistration::from_config` the session did, and
`from_config`'s anchor check REQUIRED the anchor to already exist on disk —
so on a genuinely fresh host (`artifact_dir` not yet created) `load_from`
refused, while `InferenceSession::new`/`open` on the IDENTICAL config
silently accepted it, because `JammiSession::new`'s own catalog open
(`crates/jammi-db/src/session.rs::Catalog::open_with_tenant`'s
`std::fs::create_dir_all(artifact_dir)`) had already created the anchor as a
side effect BEFORE `wrap_with`'s `from_config` call ever ran — two
enforcement points, two different verdicts on one config. The fix splits the
check:

`crates/jammi-db/src/catalog/instance.rs` (`MembershipConfig::validate`,
`:277-348`) is PURE — no filesystem read/write of any kind:

```
peer_advertise unset  => Ok(None)                                          // non-member
peer_advertise = addr => peer_bind must be Some, else JammiError::Config
                         naming both keys
                      => PeerAddr::parse(addr)?
                      => the effective root's scheme parses, is not
                         memory://, and (file:// only) its anchor is
                         ABSOLUTE — else JammiError::Config naming the key (F2)
                      => Ok(Some(MembershipConfig { peer_addr, effective_root }))
```

`JammiConfig::load_from` (`crates/jammi-db/src/config/mod.rs:2537-2597`)
calls ONLY this (`:2595`), discarding the `Option` — loading a config file
must never itself create a directory as a side effect, so a bad
`peer_advertise`/`peer_bind` pair or a RELATIVE anchor is refused at load,
naming the key, while a merely-MISSING (but absolute, well-formed) anchor is
ACCEPTED — the pure check has nothing to say about existence at all.

`InstanceRegistration::from_config` (`:210-228`) = `MembershipConfig::
validate` PLUS `MembershipConfig::materialize` (`:358-411`): for a `file://`
anchor, `std::fs::create_dir_all`s it if absent (idempotent with
`JammiSession`'s own `create_dir_all` of `artifact_dir`, and with
`ResultStore`'s later one of the same `result_root` path — refusing naming
the key if it exists as a non-directory or cannot be created), then
`std::fs::canonicalize`s it and applies the leaf rule; a cloud root needs no
filesystem step. This is the ONLY constructor `Catalog::upsert_instance` /
`Catalog::reregister_instance` accept (enforced by their own signatures,
§1.3 — both take `&InstanceRegistration`, never raw fields).

`crates/jammi-ai/src/session.rs::InferenceSession::wrap_with`
(`crates/jammi-ai/src/session.rs:207-395`) calls `from_config` at
`crates/jammi-ai/src/session.rs:229-238`, BEFORE the lease keeper starts
(`:254-261`), before the result store creates any directory (`:272-275`),
and before any prune/reclaim sweep (`:329-361`) — but `wrap_with` receives an
ALREADY-CONSTRUCTED `JammiSession` (`inner`, its parameter), whose catalog
connection is already open and whose `artifact_dir` the caller's own
`JammiSession::new` already created; a `from_config` refusal here returns
`Err` without closing that connection, so "runs before wrap_with's own
side effects" is not the same claim as "leaves nothing behind" (§2.8b A2 in
`docs/maintainer/MAINTAINER-GUIDE.md` restates this precisely; `jammi-ai`'s
`tests/it/instance_identity.rs::file_result_root_anchor_fails_open_and_writes_no_row`
reopens the SAME catalog directory in a bounded retry loop after a
deliberately-failed construction for exactly this reason). `wrap_with` is
the universal funnel: every `InferenceSession` constructor (`new`, `open`,
`open_with_placement`) reaches it, confirmed by
`crates/jammi-ai/src/session.rs:128,149,200`, all of which call `wrap_with`
either directly or through `wrap`.

`ServerConfig::validate` (`crates/jammi-db/src/config/mod.rs:1991-2041`,
the 3-way listener-collision check) is NOT the home for either half: it has
no access to `artifact_dir`/`storage.result_root`, which the anchor check
needs (`config: &JammiConfig`, `MembershipConfig::validate`'s own doc,
`crates/jammi-db/src/catalog/instance.rs:263-276`, and
`crates/jammi-db/src/config/mod.rs:2581-2591`).

**P-B2 purity, restated.** Given an ABSOLUTE anchor (guaranteed by F2's
check), `MembershipConfig::materialize`'s output depends on nothing but the
config: `crates/jammi-db/src/config/tests.rs::
canonical_result_root_is_independent_of_a_chdir_between_load_and_materialize`
proves it survives a `std::env::set_current_dir` between `validate` and
`materialize`.

### 1.3 The two read verbs and the two writers

`crates/jammi-db/src/catalog/jobs_repo.rs` (at 21a4bd0c):

- **`upsert_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2070-2102`): one `INSERT ...
  VALUES (...) ON CONFLICT(instance_id) DO UPDATE SET label, host,
  peer_addr, result_root, last_seen_at`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2081-2095`) — `started_at` is
  stamped only on the initial insert (never in the `DO UPDATE SET` list).
  `reg.peer_addr`/`reg.canonical_root` write `NULL` for a non-member
  registration.
- **`reregister_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2114-2161`): the SAME
  `instances` upsert as `upsert_instance`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2125-2142`), plus, INSIDE THE
  SAME TRANSACTION (`crates/jammi-db/src/catalog/jobs_repo.rs:2122-2159`,
  one `backend().transaction(...)` call), an `INSERT ... ON CONFLICT(instance_id)
  DO UPDATE SET kinds, state` into `workers`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2144-2154`) — but ONLY when
  `reg.worker_snapshot()` (taken once, before the transaction, at
  `crates/jammi-db/src/catalog/jobs_repo.rs:2120`) is `Some`. This is the ONLY
  verb that re-creates a pruned row; `touch_instance`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2165-2181`, a pure `UPDATE ...
  WHERE instance_id = $2`) can never resurrect one (`Ok(updated == 1)` at
  `:2180`, `false` when the row is absent).
- **`fresh_instance(instance_id, lease)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2190-2218`, unchanged from base
  except its doc): `true` iff a row is present AND NOT
  `stale_before_clause("last_seen_at", kind, margin, ...)`, where `margin =
  instance_liveness_margin(lease)` (`:2193`, `super::lease`, §1.6). No tenant
  predicate — `instances` carries no tenant column
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2185-2189`).
- **`peer_addr_of(instance_id, lease) -> Result<Option<PeerAddr>>`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2235-2275`): `SELECT peer_addr
  FROM instances WHERE instance_id = $N AND peer_addr IS NOT NULL AND NOT
  ({stale})` (`:2256-2260`); `None` when the row is absent, stale, or
  `peer_addr` is NULL, ALL collapsed the same way (a single `query_opt`, no
  distinguishing return). A stored `peer_addr` that fails `PeerAddr::parse`
  is a typed `JammiError::Catalog` (`:2269-2273`), never silently mapped to
  `None` — a corrupted row fact is distinct from an absent/stale one. No
  kind/root/self filter (doc, `:2220-2228`): this is the ONE by-id
  resolution surface, used by a rank resolving its own coordinator or any
  member resolving any other by id, including a busy or other-kind one.
- **`list_gang_members(GangListing)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2298-2368`): SQL
  (`:2314-2321`) is `SELECT i.instance_id, i.peer_addr, i.result_root,
  w.kinds, w.state FROM instances i JOIN workers w ON w.instance_id =
  i.instance_id WHERE i.peer_addr IS NOT NULL AND i.result_root IS NOT NULL
  AND NOT ({stale})` — an INNER join (a member must have a `workers` row: a
  fleet worker with a claim-loop slot, doc `:2283-2284`), and freshness/
  NULL-ness pushed into SQL while everything else stays in Rust
  (`:2287-2292`). The Rust filter loop (`:2339-2365`) excludes, in order: the
  caller itself (`instance_id == listing.self_instance`, `:2340-2342`);
  `state != WorkerState::Claiming.as_db_str()` (`:2343-2345`, so
  `warming`/`draining` are excluded); `kinds.split(',').map(str::trim).any(|t|
  t == listing.kind)` false (`:2346-2352`, a WHOLE trimmed token, never a
  substring — `fine_tune` cannot match `graph_fine_tune`); `result_root.as_bytes()
  != listing.canonical_root.as_str().as_bytes()` (`:2353-2355`, a Rust byte
  comparison, never SQL `=`). A surviving row's `peer_addr` is parsed
  (`:2356-2360`, the same typed-error-on-corruption shape as `peer_addr_of`)
  and pushed as a `GangMember`. The final list is
  `members.sort_by(|a, b| a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()))`
  (`:2366`) — Rust byte-order sort, never a SQL `ORDER BY` (backend collation
  is untrusted, per the doc at `:2277-2292`).
- **`upsert_worker`** (`crates/jammi-db/src/catalog/jobs_repo.rs:2377-2404`),
  **`set_worker_state`** (`:2409-2425`), and **`delete_worker`**
  (`:2481-2496`) are unchanged in SQL shape from base; their callers moved
  (§1.5). `upsert_worker`'s own doc (`crates/jammi-db/src/catalog/jobs_repo.rs:2370-2376`)
  and `WorkerFacts::kinds`'s doc
  (`crates/jammi-db/src/catalog/instance.rs:110-111`) both still read "the
  comma-joined (or otherwise producer-encoded) kind set" verbatim — this
  phrase is NOT tightened by this unit (the verb itself performs no
  separator validation); what IS pinned is that the sole production
  producer, `JobWorker::run_until`/`EmbeddedWorker::begin_drain`
  (`crates/jammi-ai/src/fine_tune/worker.rs:822,827,851`), always calls
  `self.kinds.join(",")`, so `list_gang_members`'s comma-split-whole-token
  match (`:2346-2352` above) is exercised against a comma-joined string in
  every real deployment, never merely asserted possible.
- **`prune_instances(stale_after)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2527-2539`): `DELETE FROM
  instances WHERE {stale}` — its own `workers` row cascades
  (`ON DELETE CASCADE`, confirmed in `crates/jammi-db/src/catalog/schema.rs:1030`
  region per the source contract's citation, re-verified: `workers.instance_id`
  FK is declared `ON DELETE CASCADE` in the same migration 029 block).

### 1.4 `[server] peer_advertise` and `JammiConfig::canonical_result_root`

`crates/jammi-db/src/config/mod.rs` (at 21a4bd0c):

- **`ServerConfig::peer_advertise: Option<String>`**
  (`crates/jammi-db/src/config/mod.rs:1636`, doc `:1617-1635`): "the address
  OTHER replicas dial THIS process's `peer_bind` listener at ... `None`
  (the default) = this process never advertises a gang membership row: its
  `instances.peer_addr`/`result_root` columns stay `NULL` regardless of
  whether `peer_bind` is set. Requires `peer_bind` to be set too (refused at
  the ONE membership choke point, `InstanceRegistration::from_config` —
  naming BOTH keys)". This doc wording is itself the "refused at load"
  language corrected to name the constructor (§8 B3's binding fold, source
  contract).
- **`resolved_result_root(&self) -> Result<String>`**
  (`crates/jammi-db/src/config/mod.rs:2478-2491`): `storage.result_root`
  when `Some`, else `self.artifact_dir.join("jammi_db")`'s UTF-8 string
  (`:2480-2489`) — the SAME derivation `ResultStore::new`'s local-root arm
  performs (doc, `:2464-2469`). A1 fix: fallible — refuses naming
  `artifact_dir` when the joined path is not valid UTF-8 (`:2483-2488`),
  never `Path::to_string_lossy`'s silent replacement-character fold (which
  could make two genuinely different paths compare equal downstream).
- **`canonical_result_root(&self) -> Result<Option<CanonicalRoot>>`**
  (`crates/jammi-db/src/config/mod.rs:2516-2530`): a thin convenience
  wrapper — `MembershipConfig::validate(self)?` (`:2517`), then, if
  `Some`, `.materialize()?` (`:2519`) — over the split §1.2 states in full;
  `Ok(None)` when `peer_advertise` is unset. `resolved_result_root`'s two
  arms are still the effective root this canonicalizes: `{artifact_dir}/
  jammi_db` (the `artifact_dir` anchor, created if absent, canonicalized,
  leaf appended lexically) when `result_root` is unset, `result_root`
  VERBATIM (the SAME string `ResultStore::with_root` roots the store at —
  no suffix) when set; a cloud scheme is lowercased, folded through
  `Scheme`'s own alias table, and trailing-`/`-trimmed.
- **`JammiConfig::load_from`**'s `MembershipConfig::validate` call is at
  `crates/jammi-db/src/config/mod.rs:2595` (§1.2) — NOT `from_config`, per
  the F1 fix: the PURE half only.

### 1.5 The claim-loop's ownership of the worker half

`crates/jammi-ai/src/fine_tune/worker.rs` (at 21a4bd0c):

- **`JobWorker::run_until`** (`crates/jammi-ai/src/fine_tune/worker.rs:805-…`):
  sets the registration's worker cell to `Some(WorkerFacts { kinds:
  self.kinds.join(","), state: WorkerState::Warming })`
  (`crates/jammi-ai/src/fine_tune/worker.rs:819-824`) BEFORE issuing the
  first `upsert_worker` (`:825-831`) — the comment at `:815-818` states this
  ordering exists so a keeper reregister racing this very first upsert can
  never observe an empty cell. After the worker gate opens, the SAME
  cell-before-row order flips the cell to `Claiming`
  (`crates/jammi-ai/src/fine_tune/worker.rs:848-853`) before calling
  `set_worker_state` (`:854-860`).
- **`EmbeddedWorker`** (`crates/jammi-ai/src/fine_tune/worker.rs:2660-2698`)
  holds `registration: Arc<InstanceRegistration>`
  (`:2687`, doc `:2680-2686`), cloned from
  `session.instance_registration()` at spawn
  (`crates/jammi-ai/src/fine_tune/worker.rs:2738`) — the SAME `Arc` the
  session's own `LeaseTarget::Instance` hold renews (confirmed:
  `crates/jammi-ai/src/session.rs:370-372` holds
  `LeaseTarget::Instance(Arc::clone(&registration))` from the SAME
  `registration` variable `instance_registration: registration` is stored as
  at `:390`). `begin_drain`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2768-2791`) preserves the cell's
  `kinds`, flips only `state` to `Draining`
  (`:2776-2783`), THEN calls `set_worker_state`
  (`:2784-2790`) — cell before row, same ordering. `stop_and_join`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2831-2853`) clears the cell
  (`self.registration.set_worker(None)`, `:2850`) BEFORE
  `self.catalog.delete_worker(&self.instance_id)` (`:2851`) — cell before
  delete, once the loop task has fully returned (comment,
  `:2848-2849`), so nothing can race a re-set of the cell after the clear.
- **`InferenceSession::instance_registration()`**
  (`crates/jammi-ai/src/session.rs:473-477`) is the `pub(crate)` getter both
  `EmbeddedWorker::spawn_worker` and the lease keeper's hold construction
  (`crates/jammi-ai/src/session.rs:370-372`) read from — the SAME `Arc`
  throughout a session's lifetime.

### 1.6 The prune window and the keeper's reregister

`crates/jammi-db/src/catalog/lease.rs` (at 21a4bd0c):

- **`instance_liveness_margin(lease) -> Duration`**
  (`crates/jammi-db/src/catalog/lease.rs:287-289`): `lease.saturating_mul(2)`
  — unchanged from base (this unit only adds a second production caller,
  `fresh_instance`/`peer_addr_of`/`list_gang_members`, to the one U5a-1
  already created).
- **`instance_prune_window(lease) -> Duration`**
  (`crates/jammi-db/src/catalog/lease.rs:301-303`), NEW: `instance_liveness_
  margin(lease).saturating_add(lease)`, i.e. `3 × lease` — STRICTLY beyond
  the margin (doc, `:291-300`, and the self-test
  `prune_window_is_strictly_beyond_the_liveness_margin`,
  `crates/jammi-db/src/catalog/lease.rs:322-338`, which asserts `window ==
  3×lease` exactly and `window > margin` for `{1, 5, 30, 3600}` second
  leases, plus the `Duration::MAX` saturating-add case). The doc states the
  defect this closes explicitly (`:296-300`): before this function existed,
  the only caller (`InferenceSession::wrap_with`) pruned at exactly the
  liveness margin, so a merely-stale member was ALREADY prune-eligible the
  instant it read stale.
- **`InferenceSession::wrap_with`**'s prune call
  (`crates/jammi-ai/src/session.rs:341-345`):
  `catalog.prune_instances(jammi_db::catalog::lease::instance_prune_window(
  lease_intervals.lease())).await?` — the named function, never a literal
  `saturating_mul(2)`/`(3)` at the call site.
- **`renew_all`'s `Instance` arm**
  (`crates/jammi-db/src/catalog/lease_keeper.rs:772-797`):
  `touch_instance(&reg.instance_id)` → `Ok(true)` renews (stamps
  `last_renewed_ms`, `:851-852`); `Ok(false)` (a missed touch — the row was
  pruned during a transient outage, per the comment at `:774-779`) calls
  `catalog.reregister_instance(reg)` INSTEAD of flipping `lost`
  (`:780-789`) — `Ok(())` is treated as renewed (`Some(true)`), an `Err` warns
  and neither renews nor flips `lost` (`None`, `:781-789`, same shape as a
  faulted `touch_instance`, `:790-796`). The `Some(true)`/`Some(false)`/`None`
  dispatch (`crates/jammi-db/src/catalog/lease_keeper.rs:851-855`) confirms:
  `None` is a genuine no-op, never a silent `lost` flip on a keeper-side
  fault.
- **`LeaseTarget::Instance(Arc<InstanceRegistration>)`**
  (`crates/jammi-db/src/catalog/lease_keeper.rs:58-67`) carries the `Arc`, not
  an owned copy — doc `:63-66`: the SAME registration `JobWorker` mutates in
  place is the one this hold renews, so a state change lands on the very
  next renewal with no re-registration.

---

## 2. Properties (quantified, never a single-input claim)

**P-M3 (the membership predicate, restated over the shipped shape).** For
every `list_gang_members(GangListing { kind, self_instance, canonical_root,
lease })` call and every DB row order: a row is returned iff it is NOT
`self_instance`, has a `workers` row (INNER join) with `state == Claiming`,
`kinds` contains `kind` as a whole trimmed comma-split token (a kind that is
merely a SUBSTRING of a token, e.g. `fine_tune` vs. `graph_fine_tune`, is
excluded — `crates/jammi-db/tests/it/gang_membership.rs:272-300`), `peer_addr`
and `result_root` are both non-NULL, `result_root` bytes equal
`canonical_root` bytes (a root divergent only by case or a trailing `/` is
excluded — `crates/jammi-db/tests/it/gang_membership.rs:307-347`), and it is
fresh under `instance_liveness_margin(lease)`; the surviving set is sorted by
`instance_id` byte order regardless of the underlying row order (proven
against a DESCENDING-insertion-order fixture with a raw-SELECT vacuity
control, `crates/jammi-db/tests/it/gang_membership.rs:531-604`). `peer_addr_of`
resolves a busy or other-kind fresh member (no kind/state filter) and is
`None` for a stale or NULL-`peer_addr` one
(`crates/jammi-db/tests/it/gang_membership.rs:656-756`). Both verbs return
the SAME answer under a scoped tenant binding and under none
(`crates/jammi-db/tests/it/gang_membership.rs:606-645`, mirroring
`Catalog::get_job_for_rank`'s own tenant-independence oracle).

**P-M4 (recovery restores the WHOLE membership tuple).** For a process that
IS a gang member (a non-NULL `peer_addr`/`result_root` row AND a `workers`
row with `state == Claiming`), force-deleting its `instances` row and running
one real lease-keeper pass makes `list_gang_members` return it again, with
`peer_addr`, `result_root`, `kinds`, and `state` byte-identical to before
the delete — proven twice, at the catalog layer with a hand-built keeper
(`crates/jammi-db/tests/it/gang_membership.rs:762-827`,
`keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete`) and at
the session layer through the session's OWN keeper, no hand-built one
(`crates/jammi-ai/tests/it/instance_identity.rs:434-489`,
`a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass`). A
`fresh_instance`-only assertion is NOT this oracle — the design contract's
own binding fold (§8 B1) states this explicitly, and this unit's oracles
assert the `workers` row's `kinds`/`state` too, not merely the `instances`
row's presence. The dual is proven too: a DRAINED worker (its registration
cell cleared, its `workers` row deleted) is NOT resurrected as a member after
a forced delete of its `instances` row — the keeper reregisters the process
(it is still alive) but never re-inserts a `workers` row, so the INNER join
still excludes it
(`crates/jammi-ai/tests/it/instance_identity.rs:491-551`,
`a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`).

**P-prune-window.** A member merely stale in `(instance_liveness_margin(lease),
instance_prune_window(lease)]` is NOT pruned; one strictly beyond the window
IS — proven at the catalog layer directly
(`crates/jammi-db/tests/it/gang_membership.rs:836-876`,
`prune_window_does_not_prune_a_member_merely_stale_within_the_window`) and
through the REAL construction sweep, where only the RIGHT function
(`instance_prune_window`, `3×lease`) — never the old literal
`saturating_mul(2)` (which equals the margin itself) — can leave such a row
standing (`crates/jammi-ai/tests/it/instance_identity.rs:553-591`,
`a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep`).

**P-M5 (over the constructor, restated).** `peer_advertise` without
`peer_bind` → typed error naming both keys
(`crates/jammi-db/src/config/tests.rs:3373-3397`, and at `load_from`
directly, `:3314-3329`,
`crates/jammi-ai/tests/it/instance_identity.rs:366-392`, through the REAL
`InferenceSession::new`); a RELATIVE anchor (F2, the sibling fix in this
same commit) → typed error naming the key, both `artifact_dir` and an
explicit `result_root`, PURE (no filesystem access)
(`crates/jammi-db/src/config/tests.rs:3242-3274`, and at `load_from`
directly, `:3331-3352`); a MISSING (but absolute, well-formed) anchor →
CREATED, never refused — F1's own fix, replacing the "must already exist"
sentence this contract stated before this commit
(`crates/jammi-db/src/config/tests.rs:3399-3418,3580-3599`, and at
`load_from` directly, `:3354-3371`, which additionally asserts the
directory is NOT created — the pure half never touches the filesystem —
`crates/jammi-ai/tests/it/instance_identity.rs:394-423`, through the REAL
`InferenceSession::new`, which DOES materialize it); a non-directory
(FILE) anchor → refused naming the key, for both `artifact_dir` and an
explicit `result_root`
(`crates/jammi-db/src/config/tests.rs:3420-3430,3601-3617`,
`crates/jammi-ai/tests/it/instance_identity.rs:472-515`, which additionally
reopens the SAME catalog directory after the failed construction and asserts
ZERO `instances` rows — the failed check runs before any write); `result_root`
UNSET → accepted, canonicalizing `{artifact_dir}/jammi_db`
(`crates/jammi-db/src/config/tests.rs:3432-3446`,
`crates/jammi-ai/tests/it/instance_identity.rs:316-332`, through the REAL
`InferenceSession::open`); `result_root` SET (an existing directory) →
accepted, canonicalizing THAT root VERBATIM, no `jammi_db` leaf
(`crates/jammi-ai/tests/it/instance_identity.rs:340-361`, through
`InferenceSession::open_with_placement`); a library config (no
`peer_advertise`) → NULLs, never an error
(`crates/jammi-db/src/config/tests.rs:3448-3459`,
`crates/jammi-ai/tests/it/instance_identity.rs:303-309`, through a real
session). **F1's own oracle** — the two enforcement points reach the SAME
verdict on one config —
(`crates/jammi-ai/tests/it/instance_identity.rs:425-470`,
`load_from_and_session_open_agree_on_a_fresh_artifact_dir`) drives the REAL
public `JammiConfig::load_from` (over a real TOML file on a fresh
`artifact_dir`) AND `InferenceSession::new` on the SAME config, asserting
`load_from` creates nothing and the session writes a non-NULL member row.

**P-B2 (the canonical root is a pure function of the config, GIVEN the
anchor).** `canonical_result_root()` returns the identical string whether
the default `jammi_db` leaf is absent, present, or a symlink to elsewhere
(`crates/jammi-db/src/config/tests.rs:3461-3493`); two spellings of one
anchor (`./`, a trailing `/`, a doubled `/`) fold to the identical string
(`crates/jammi-db/src/config/tests.rs:3495-3535`); the canonical string
equals `canon ∘ resolved_result_root()` for BOTH arms
(`crates/jammi-db/src/config/tests.rs:3537-3578`); a cloud scheme is
lowercased and alias-folded, with a trailing `/` trimmed
(`crates/jammi-db/src/config/tests.rs:3630-3654`); `memory://` is refused
for a gang member (`crates/jammi-db/src/config/tests.rs:3619-3628`). The
property SURVIVES a `chdir` between load and materialize, given the
absolute anchor F2 guarantees
(`crates/jammi-db/src/config/tests.rs:3276-3312`,
`canonical_result_root_is_independent_of_a_chdir_between_load_and_materialize`
— guards its own `std::env::set_current_dir` window with a local `Mutex`
since cwd is process-global).

---

## 3. Oracles, by name, and what each EXCLUDES

| Oracle | Asserts | Excludes |
|---|---|---|
| `list_excludes_the_caller_itself` (`crates/jammi-db/tests/it/gang_membership.rs:138-165`) | The self row is never returned even though it is otherwise a valid member | Does not test a self row that is ALSO stale/wrong-kind (isolates the self-exclusion conjunct alone) |
| `list_excludes_a_stale_member` (`:172-199`) | A row past the liveness margin is excluded | Does not probe the exact margin boundary — that is `gang_instance_freshness.rs`'s job |
| `list_excludes_a_draining_worker` (`:206-231`) / `list_excludes_a_warming_worker` (`:238-265`) | `state != Claiming` excludes, for BOTH non-claiming states independently | Neither combines a non-claiming state with a second false conjunct |
| `list_excludes_a_kind_that_is_only_a_substring_token` (`:272-300`) | `fine_tune` never matches a `graph_fine_tune` token | Does not test a kind that is a PREFIX/SUFFIX in the other direction (the split-and-exact-match shape makes the direction irrelevant, but only one direction is fixtured) |
| `list_excludes_a_root_divergent_by_case_or_trailing_slash` (`:307-346`) | A byte-divergent root (case, trailing `/`) is excluded — proving the comparison is byte-exact, never a normalized/case-insensitive one | Does not test a root divergent by a symlink or a `..`-equivalent path (that is `canonical_result_root`'s own canonicalization job, upstream of this comparison) |
| `list_excludes_a_null_peer_addr` (`:353-383`) | `peer_addr IS NULL` excludes even with a fresh, correctly-kinded, claiming, root-matching row otherwise | Does not test `result_root IS NULL` in the same row (next oracle isolates it) |
| `list_excludes_a_member_with_peer_addr_set_but_result_root_null` (`:390-447`) | The asymmetric NULL case (migration 035 has no paired CHECK, so this row IS representable) is still excluded | Does not test the reverse (`result_root` set, `peer_addr` NULL) — the SQL predicate is symmetric (`AND` of both `IS NOT NULL`), so this direction stands for both |
| `list_excludes_an_instance_with_no_workers_row` (`:454-487`) | The INNER join excludes an `instances` row with no `workers` row at all (not merely a non-claiming one) | Does not test a `workers` row for a DIFFERENT instance_id that happens to collide on kind (join key is exact) |
| `list_includes_a_fresh_multi_kind_claiming_worker` (`:494-524`) | The positive control: every conjunct held DOES return the member — proving the exclusions above are not vacuously satisfied by a fixture that could never pass anyway | Does not vary which of the two-or-more kinds is queried (asserts inclusion for one of them) |
| `list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order` (`:531-604`) | Byte-order sort holds even when the raw SELECT (forced off a covering-index scan by selecting a second, non-indexed column) returns descending-insertion order — with an explicit control that the RAW order is asserted NOT already ascending, so the sort assertion is not vacuous | Does not test the postgres planner's own default order (both backends run this fixture; the control makes the backend's actual order irrelevant to the claim) |
| `list_gang_members_is_identical_under_a_scoped_tenant_and_under_none` (`:609-645`) | Tenant-independence, mirroring `get_job_for_rank`'s own oracle | Does not test under `with_admin_scope` (no admin-scope predicate exists on this path to begin with, unlike the gang RPC's) |
| `peer_addr_of_resolves_a_busy_or_other_kind_fresh_member` (`:651-680`) | No kind/state filter on this verb — a draining, other-kind row still resolves by id | Does not test a `NULL` `peer_addr` row (next oracle) |
| `peer_addr_of_is_none_for_a_stale_instance` / `_for_a_null_peer_addr` / `_for_an_absent_instance` (`:681-756`) | Three independent `None`-producing causes, each isolated | Does not distinguish the three by return shape — that IS the property (all `None`) |
| `keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete` (`:757-827`) | P-M4 at the catalog layer: `peer_addr` byte-identical via `peer_addr_of` too (not only via `list_gang_members`), after a force-delete and one real keeper pass | Does not exercise the session-construction path (next file does) |
| `prune_window_does_not_prune_a_member_merely_stale_within_the_window` (`:831-877`) | Both boundary directions: `(margin, window]` survives, past-window is pruned — a symmetric before/after assertion in one test | Does not test EXACTLY at the window boundary (only strictly inside vs. strictly past) |
| `library_config_without_peer_advertise_writes_null_membership_columns` (`crates/jammi-ai/tests/it/instance_identity.rs:302-309`) | A real session with no `peer_advertise` writes NULL/NULL | Does not test a session that LATER sets `peer_advertise` (no such runtime mutation path exists — config is fixed at session construction) |
| `peer_advertise_set_result_root_unset_produces_a_nonnull_row_via_open` (`:315-332`) | Arm (a) through `InferenceSession::open`: the row's `result_root` equals `canonical_result_root()`'s own value | Does not test `open_with_placement` (next oracle) |
| `peer_advertise_set_result_root_set_produces_a_nonnull_row_via_open_with_placement` (`:339-361`) | Arm (b) through `open_with_placement`, and explicitly asserts NO `jammi_db` leaf in the written value (the B2-erratum arm) | Does not test a cloud-scheme `result_root` through this real-construction path (config/tests.rs covers cloud schemes at the pure-function layer only) |
| `peer_advertise_without_peer_bind_fails_open_naming_both_keys` (`:365-381`) | The typed-error path through `InferenceSession::new`, not merely the pure `from_config` unit test | Does not test the SAME failure through `open`/`open_with_placement` (all three funnel through the same `wrap_with`, so one is representative) |
| `missing_result_root_anchor_is_created_and_session_open_succeeds` (`:394-423`) | F1's fix: a missing (but absolute, well-formed) anchor is CREATED, never refused — session open succeeds and the row is non-NULL | Does not test a non-directory (vs. missing) anchor through this real-construction path (`file_result_root_anchor_fails_open_and_writes_no_row`, next but one, covers that arm) |
| `load_from_and_session_open_agree_on_a_fresh_artifact_dir` (`:425-470`) | F1's own oracle, stated directly: `JammiConfig::load_from` on a fresh anchor succeeds AND creates nothing, and `InferenceSession::new` on the SAME config materializes it and writes the non-NULL row — the two enforcement points agree | Does not drive `load_from`'s file-resolution search path (`JAMMI_CONFIG`, `./jammi.toml`, …) — an explicit path is passed, matching every other test in this file |
| `file_result_root_anchor_fails_open_and_writes_no_row` (`:472-515`) | The ONE case a missing/creatable anchor can never be confused with: an existing FILE anchor is refused naming the key, and the row is NEVER written — proven by reopening the SAME catalog directory and counting `instances` rows | Does not test the same anchor shape for `artifact_dir` (unset `result_root`) through this real-construction path (config/tests.rs covers both anchors at the pure-function layer) |
| `a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass` (`:517-577`) | P-M4 through the session's OWN keeper (never a hand-built one), with a real `EmbeddedWorker` | Does not test a session with NO worker spawned (the drained-worker oracle, next, covers the no-`workers`-row-after-recovery case) |
| `a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete` (`:579-641`) | The `instances` row DOES reregister (the process is alive) but `workers` does NOT re-appear — the INNER join's exclusion holds even after a real recovery | Does not test a re-spawn of a NEW worker after the drain (a fresh `upsert_worker` would naturally re-include it; this oracle is about the drained state persisting through ONE recovery, not about a subsequent claim-loop restart) |
| `a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep` (`:643-684`) | The construction-sweep prune call uses `instance_prune_window`, not the old `saturating_mul(2)` literal, through a REAL second session's boot — A3 fix: a longer (9 s) lease and an `ago` biased a quarter of the `(margin, window]` gap past `margin`, widening the real-wall-clock slack before the boot sweep runs from ~1.5 s to ~6.75 s so it is never red under load | Does not test the symmetric past-window-is-pruned case through this real path (the catalog-layer oracle above covers both directions; this one isolates the regression the function extraction fixes) |
| `migration_035_is_ordered_after_034_and_adds_instances_peer_addr_result_root` (`crates/jammi-db/tests/it/migrations.rs:2012-2142`) | Ordering (relative position, never `.last()`); both columns nullable on both dialects; a both-NULL row and a both-set row are both valid inserts (no paired CHECK, unlike migration 034) | Does not test a row with exactly one of the two set through THIS oracle — `list_excludes_a_member_with_peer_addr_set_but_result_root_null` (gang_membership.rs) covers that shape's LISTING behavior, not the schema's acceptance of it (which this oracle's own both-null/both-set inserts imply by omission: no CHECK exists to reject the mixed case, and no test asserts the mixed insert fails, which would be the wrong assertion) |
| `migration_029_copies_training_jobs_rows_into_jobs_as_queued` (`crates/jammi-db/tests/it/migrations.rs:823-988`) | The fourth K5 pin site gets teeth: `035` is on the ledger's manufactured-pre-029 DELETE list, so the reopened, replayed `instances` table carries both new columns (`:958-987`) | Proves the pin site's ABSENCE would be RED (an omission leaves the columns missing) — does not itself remove the entry to observe the red; that was done once at authorship time per the source contract's own methodology precedent, not re-executed by this file |
| `instance_liveness_margin_is_twice_the_lease` / `prune_window_is_strictly_beyond_the_liveness_margin` (`crates/jammi-db/src/catalog/lease.rs:309-320,322-338`) | The `2×`/`3×` factors and the `Duration::MAX` saturating-arithmetic edge for both functions | Does not test a lease of zero (a lease is validated positive upstream, `LeaseConfig`'s own validation, not re-proven here) |
| `registration_worker_half_starts_unpopulated_and_is_settable` (`crates/jammi-db/src/catalog/instance.rs:488-500`) | The cell's own set/clear/snapshot round-trip in isolation | Does not test concurrent set/snapshot from two threads (the `Mutex` makes this a liveness, not a correctness, concern — untested here) |
| `peer_addr_parses_host_port` / `_keeps_ipv6_host_intact` / `_refuses_no_colon` / `_refuses_empty_host` / `_refuses_zero_port` / `_refuses_non_numeric_port` / `_refuses_out_of_range_port` (`crates/jammi-db/src/catalog/instance.rs:446-486`) | Every `PeerAddr::parse` edge, including the IPv6-colon-preservation shape (`rsplit_once`) | Does not test a hostname requiring DNS resolution (this type never resolves — a wire-form validator only, per its own doc) |

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
  `list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order`'s
  raw-order control is built to catch: the control proves the RAW select is
  not already sorted, so if the Rust sort were removed the final assertion
  (ascending byte order) would fail loudly on the very fixture engineered to
  defeat a covering-index scan.
- **`instance_prune_window` vs. the old literal.** Reverting
  `crates/jammi-ai/src/session.rs:342-344` to
  `lease_intervals.lease().saturating_mul(2)` (the pre-unit shape) is exactly
  what `a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep` is
  built to catch — the fixture's row is seeded stale at
  `margin + (window - margin) / 4` (A3: a QUARTER of the gap past `margin`,
  not the midpoint — widening real-wall-clock slack before `window`) so the
  old literal (equal to the margin) would prune it and the new function (the
  window) would not.
- **`reregister_instance`'s worker-half conditional.** Removing the `if let
  Some(w) = worker { ... }` guard at
  `crates/jammi-db/src/catalog/jobs_repo.rs:2155-2166` (always writing a
  `workers` row regardless of the cell) would flip
  `a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`'s
  expected "no workers row after recovery" to a false resurrection — the
  drained state (cell cleared) is the fixture that makes this conjunct
  observable.
- **The choke-point ordering.** Moving `from_config`'s call in `wrap_with`
  (`crates/jammi-ai/src/session.rs:229-238`) to AFTER `build_result_store`
  (`:272-275`) would let a FILE-anchor refusal occur only after
  `build_result_store` had ALREADY materialized the wrong thing at that
  path (a `jammi_db` file where a directory belongs, corrupting a later
  reader) rather than being caught first, on the config AS GIVEN — this
  fix's own F1 oracle no longer distinguishes "missing" from "ordering"
  (a missing anchor is accepted either way, since `MembershipConfig::
  validate` never checks existence and `from_config`'s materialize
  `create_dir_all`s it regardless of ordering); the ordering claim now
  rests on the FILE-anchor case alone —
  `file_result_root_anchor_fails_open_and_writes_no_row` is built on the
  CURRENT ordering and does not itself re-probe the ordering directly (it
  asserts the OUTCOME: zero rows after failure), so this is a documented,
  not executed, mutation.

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

---

## 6. Report

`impossibility_claims`:
- "no caller other than `InstanceRegistration::from_config` builds a value
  `Catalog::upsert_instance`/`Catalog::reregister_instance` accept" — not
  independently enumerated by a code-scanning oracle in this unit (unlike
  U5a-1's `get_job_for_rank` oracle); the claim rests on the two verbs'
  own signatures (`&InstanceRegistration`, a type with no public field
  access to `peer_addr`/`canonical_root` outside the crate — both are
  private-ish by convention, constructible only via `InstanceRegistration::new`
  or `::from_config`) rather than an executed grep-shaped enumeration —
  `uncovered` by this unit's own test suite; a future unit adding a second
  production call site of either verb would not be mechanically caught the
  way U5a-1's `get_job_for_rank` oracle catches a second caller of that verb.
- "a stored `peer_addr` that fails to parse is never silently treated as
  absent" — executed attempt: both `peer_addr_of`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2267-2274`) and
  `list_gang_members` (`crates/jammi-db/src/catalog/jobs_repo.rs:2356-2360`)
  return a typed `JammiError::Catalog` on a parse failure rather than mapping
  it to `None`/exclusion — verified by direct read of both call sites; no
  test manufactures a corrupted `peer_addr` row to observe the error
  (`uncovered`: the typed-error path itself is read-verified, not
  test-exercised, in this unit's own suite).
- "the `,` kind-encoding convention is honored by every real writer" —
  executed check: `crates/jammi-ai/src/fine_tune/worker.rs:822,827,851`
  are the ONLY three call sites of `self.kinds.join(",")` feeding
  `upsert_worker`/`WorkerFacts.kinds` in this crate (confirmed by direct
  read of the file, not by a code-scanning oracle — `uncovered` as a
  standing tripwire the way U5a-1's enumerating-caller oracle is one).

`citations_reanchored`: every citation in this document was read directly
against `21a4bd0c` in this worktree by the writing agent. `ci/scripts/perf/
check_citations.py` and the lead's `recite.py` were both run over this file
before it was finalized; every `STALE` hit `recite.py` reported was reviewed
and hand-corrected by re-reading the cited file at HEAD (never carried
forward from the source design contract's own, differently-based, line
numbers).

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
