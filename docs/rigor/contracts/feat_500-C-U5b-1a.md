# CONTRACT — feat/500-C-U5b-1a: gang-membership substrate

**Contract of record.** slug: `feat_500-C-U5b-1a` · branch `feat/500-C-U5b-1a`
at `686d4aedfefb3693ee1b798a8749f63493df1acc` (the code commit, tagged
throughout this revision as "commit 1") · this file is the committed
mechanism contract `ci/scripts/check_rigor_record.py` requires under
`docs/rigor/contracts/**` (its check 3) before this unit's rigor record at
`docs/rigor/feat_500-C-U5b-1a.jsonl` (the lead's export) satisfies check 1/2.
Source design contract: the lead's `CONTRACT-U5b-1a.md` v4 (scratchpad-only,
never a repo path — it is not cited with a `path:line` token anywhere in this
document, since it is not a tracked file at HEAD).

**P-X4 hardening (this revision, re-anchors "commit 1").** The prior commit
1 (`bd21b79081935c942bbb65154f43ee09a790aa3f`) left `MemberRoot::new` `pub`
with no `cfg` gate: P-X4's claim ("constructible in production ONLY from
`resolved_result_root()`") held BY CONVENTION (the only call site in that
diff was `from_config`) but was not ENFORCED — nothing stopped a future
in-tree caller from reaching the arbitrary-string constructor directly, a
greenfield-threat-model gap (`pub` is not a control the moment a second
in-tree caller can compile against it). "Commit 1" is re-anchored to
`686d4aedfefb3693ee1b798a8749f63493df1acc`, which adds `MemberRoot::
resolved(&JammiConfig) -> Result<Self>` as the ONE production constructor
(it calls `resolved_result_root` itself), gates `MemberRoot::new` behind
`feature = "test-hooks"`, and adds the enumerating oracle P-X4 now cites.
Every `path::construct`/`path:line` citation below was re-derived directly
against THIS commit's tree (self-check, §11).

**Round-3 revision.** This is the THIRD mechanism revision of this contract.
Rounds 1–2 built, then twice patched, a canonicalization step over the
membership root (`CanonicalRoot`, `EffectiveRoot`, `MembershipConfig::
materialize`); a third closing-audit BLOCK on that same mechanism fired the
source contract's pre-committed round-3 stop rule (`CONTRACT-U5b-1a.md`
§9's binding clause), so canonicalization is EXCISED rather than patched a
fourth time. **§1–§5 below describe the mechanism as it EXISTS at commit 1 —
CURRENT-STATE only; §6 is the history of rounds 1–3, in prose, and cites no
construct that commit 1 deleted** (a citation for a construct commit 1
removed would fail this revision's own machine-check, §7).

**Citation convention (round-2 addendum, unchanged).** A whole FUNCTION or
TYPE is cited BY CONSTRUCT — `` `path/to/file.rs::Type::method` `` or
`` `path/to/file.rs::Type` `` — never by a line range, so an unrelated edit
elsewhere in the same file can never leave the citation stale.
`path:line`/`path:a-b` is reserved for STATEMENT-level claims: a specific SQL
string, a specific conjunct/`continue` arm, a specific doc-comment sentence
being quoted. Every citation in this revision (§1–§5, §7) was re-derived
directly against the tree of commit 1 and machine-checked (§7); §6 (history)
cites no construct at all, by design.

## Scope

U5b-1a lands the gang-membership substrate DESIGN.md § 4 and DIST § 5.8
sketch: two new nullable columns on `instances` (`peer_addr`, `result_root`),
the ONE registration carrier every writer of the row builds
(`InstanceRegistration`) and its ONE choke point (`from_config`), the two
read verbs (`peer_addr_of`, `list_gang_members`), and the lease keeper's
whole-tuple reregister on a missed heartbeat. As of commit 1 (the round-3
excision), `instances.result_root` carries the VERBATIM,
byte-for-byte output of `JammiConfig::resolved_result_root()` — no
canonicalization, no filesystem access, no scheme/symlink/case
interpretation of any kind on this path. It does **not** build a caller of
either read verb (that is U5b-1b-ii's coordinator), the attestation VERIFY
that gives root-equality sufficiency (U5a-1's whole-artifact sidecar /
U5b-0's and U5b-1b-i's per-partition inventory already exist as separate
mechanisms; this unit only states that root equality alone is never
sufficient), or a spelling-identity predicate (filed as unit U5b-1a-A2, §6).

---

## 1. The carrier: `PeerAddr`, `MemberRoot`, `WorkerFacts`, `InstanceRegistration`

`crates/jammi-db/src/catalog/instance.rs`, a module.

- **`PeerAddr`** (`crates/jammi-db/src/catalog/instance.rs::PeerAddr`) is a
  sealed wrapper (`pub struct PeerAddr(String)`, private field, declared
  `crates/jammi-db/src/catalog/instance.rs:37`): the only way to build one is
  `PeerAddr::parse` (`crates/jammi-db/src/catalog/instance.rs::PeerAddr::
  parse`, `crates/jammi-db/src/catalog/instance.rs:45`), which
  `rsplit_once(':')`s the input so an IPv6 literal's own colons stay inside
  the host segment and only the LAST colon splits host from port, requires a
  non-empty host, and a port that parses as `u16` and is non-zero. `as_str()`
  and `Display` (`crates/jammi-db/src/catalog/instance.rs:73`) both return
  the validated wire form. This is the SAME type
  `crates/jammi-db/src/index/peer.rs` re-exports (per this module's own doc
  comment) — the peer listener and the gang listener dial the identical
  address type, so they can never drift into two.
- **`MemberRoot`** (`crates/jammi-db/src/catalog/instance.rs::MemberRoot`,
  declared `crates/jammi-db/src/catalog/instance.rs:105`) wraps a string
  VERBATIM — no canonicalization, no interpretation. `Self::resolved`
  (`crates/jammi-db/src/catalog/instance.rs::MemberRoot::resolved`,
  `crates/jammi-db/src/catalog/instance.rs:114`) is the ONE production
  constructor — it calls `resolved_result_root` itself, so a `MemberRoot`
  can never carry a string the resolver did not produce;
  `InstanceRegistration::from_config` (§2) is its only caller. `Self::new`
  (`crates/jammi-db/src/catalog/instance.rs::MemberRoot::new`,
  `crates/jammi-db/src/catalog/instance.rs:127`) wraps an arbitrary string
  with no resolver call at all and is compiled ONLY under
  `feature = "test-hooks"` — a real `cfg` gate, not `#[doc(hidden)]` — so it
  cannot link into a production build at all; fixtures/tests build a
  `MemberRoot` through it directly
  (`crates/jammi-db/tests/it/gang_membership.rs`,
  `crates/jammi-ai/tests/it/instance_identity.rs`,
  `crates/jammi-db/src/config/tests.rs`). Its doc states the design
  property directly: two DIFFERENT spellings of one physical location
  (`gcs://b/p` vs `gs://b/p`, a trailing `/`, a case difference) are two
  DIFFERENT roots.
- **`WorkerFacts`** (`crates/jammi-db/src/catalog/instance.rs::WorkerFacts`,
  declared `crates/jammi-db/src/catalog/instance.rs:152`) is the claim-loop
  half of a registration: `kinds: String` (the comma-joined kind set) and
  `state: WorkerState`. Its doc states its SOLE owner is `JobWorker`/
  `EmbeddedWorker` (`crates/jammi-ai/src/fine_tune/worker.rs`), confirmed at
  §4 below.
- **`InstanceRegistration`**
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration`, declared
  `crates/jammi-db/src/catalog/instance.rs:173`): `instance_id`, `label`,
  `host`, `peer_addr: Option<PeerAddr>`, `member_root: Option<MemberRoot>`,
  and `worker: Mutex<Option<WorkerFacts>>` — the claim-loop cell, mutated in
  place by its single owner rather than requiring a whole new registration
  per state change. `InstanceRegistration::new`
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::new`,
  `crates/jammi-db/src/catalog/instance.rs:191`) is the plain constructor
  with no validation, used by `from_config` itself and by fixtures/tests
  (`crates/jammi-db/tests/it/gang_membership.rs`'s `seed_member` helper and
  direct-construction call sites, and
  `crates/jammi-ai/tests/it/instance_identity.rs`'s foreign-row fixture).
  `set_worker`/`worker_snapshot` are the cell's read/write pair
  (mutex-poisoning tolerant via `unwrap_or_else(|p| p.into_inner())`).
- **`GangListing<'a>`** (`crates/jammi-db/src/catalog/instance.rs::
  GangListing`, `crates/jammi-db/src/catalog/instance.rs:294`) and
  **`GangMember`** (`crates/jammi-db/src/catalog/instance.rs::GangMember`,
  `crates/jammi-db/src/catalog/instance.rs:313`) are `list_gang_members`'s
  request/response shapes (§3).

## 2. The membership check: `MembershipConfig::validate`, `InstanceRegistration::from_config`, `resolved_result_root`

**The round-3 design, in one paragraph.** `MembershipConfig::validate`
(`crates/jammi-db/src/catalog/instance.rs::MembershipConfig::validate`,
`crates/jammi-db/src/catalog/instance.rs:278-288`) is PURE and checks
EXACTLY two things: `[server] peer_advertise` parses as a `PeerAddr`
(`:286`), and `[server] peer_bind` is set too, else a typed
`JammiError::Config` naming BOTH keys (`:282-284`). It inspects
`result_root`/`artifact_dir` NOT AT ALL. `InstanceRegistration::from_config`
(`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::from_config`,
`crates/jammi-db/src/catalog/instance.rs:239-255`) runs `validate`; when
membership applies, it sets `member_root` to
`MemberRoot::resolved(config)?` (`:248`), which is itself
`Self(config.resolved_result_root()?)` — the SAME string
`build_result_store` (`crates/jammi-ai/src/session.rs::build_result_store`,
`crates/jammi-ai/src/session.rs:2418`) hands to `StorageUrl::parse` before
rooting the `ResultStore` there. `JammiConfig::resolved_result_root`
(`crates/jammi-db/src/config/mod.rs::JammiConfig::resolved_result_root`,
`crates/jammi-db/src/config/mod.rs:2481-2492`) is the ONE effective-root
derivation (`storage.result_root` when set, else
`self.artifact_dir.join("jammi_db")`'s UTF-8 string) and the ONLY function on
the whole membership path that can fail — it refuses naming `artifact_dir`
when the joined default-arm path is not valid UTF-8
(`crates/jammi-db/src/config/mod.rs:2486-2489`), never a silent
`to_string_lossy()` fold.

**No interpretation, anywhere on this path.** Neither `validate` nor
`from_config` opens a file, calls `std::fs::canonicalize`, parses a
`StorageUrl`, or inspects a scheme. `JammiConfig::load_from`
(`crates/jammi-db/src/config/mod.rs::JammiConfig::load_from`) calls
`MembershipConfig::validate` directly, discarding the `Option`
(`crates/jammi-db/src/config/mod.rs:2566`) — the early-failure check.
`InferenceSession::wrap_with` (`crates/jammi-ai/src/session.rs::
InferenceSession::wrap_with`) calls `from_config`
(`crates/jammi-ai/src/session.rs:231-236`), FIRST — before the lease keeper
starts, before the result store touches anything. `wrap_with` is the
universal funnel: every `InferenceSession` constructor (`new`, `open`,
`open_with_placement`) reaches it, confirmed by
`crates/jammi-ai/src/session.rs::InferenceSession::open`,
`::open_with_placement`, and the private `::wrap`, all of which call
`wrap_with` either directly or through `wrap`. `ServerConfig::validate` is
NOT the home for either check: it has no access to `artifact_dir`/
`storage.result_root`, which `resolved_result_root` needs.

**F1 narrative, corrected (round-3; the round-2 text propagated a wrong
construct).** Rounds 1–2 narrated the pre-unit defect this way: on a
genuinely fresh host, `JammiConfig::load_from` refused a not-yet-existing
`artifact_dir` while `InferenceSession::new` on the identical config silently
accepted it, because the SESSION's own catalog open had already
`create_dir_all`'d the anchor before the membership check ever ran. The
CORRECT construct for that catalog-open pre-creation, verified against
commit 1's tree, is `crates/jammi-db/src/session.rs::
build_backend_from_config`'s SQLite-default arm
(`crates/jammi-db/src/session.rs:1121-1131`, the `path: None` branch,
`std::fs::create_dir_all(&config.artifact_dir)?` at
`crates/jammi-db/src/session.rs:1127`) — NOT
`crates/jammi-db/src/catalog/mod.rs::Catalog::open_with_tenant`
(`crates/jammi-db/src/catalog/mod.rs:65-78`, its own `create_dir_all` at
`:69`), which is a real construct but is NOT on `JammiSession::new`'s
construction path at all: its only caller is `Catalog::open`
(`crates/jammi-db/src/catalog/mod.rs:57-59`), and neither is reachable from
`InferenceSession::new`/`JammiSession::new`. Under a Postgres catalog,
NOTHING on `JammiSession::new`'s path creates `artifact_dir` — the round-3
design makes this moot anyway, since `resolved_result_root`'s default arm
never requires the directory to exist (it is a string join, not a
filesystem check), so the pre-unit defect this paragraph narrates no longer
has a filesystem precondition to race.

## 3. The two read verbs and the two writers

`crates/jammi-db/src/catalog/jobs_repo.rs`:

- **`upsert_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::upsert_instance`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2082-2114`): one `INSERT ...
  VALUES (...) ON CONFLICT(instance_id) DO UPDATE SET label, host,
  peer_addr, result_root, last_seen_at`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2093-2099`) — `started_at` is
  stamped only on the initial insert. `reg.peer_addr`/`reg.member_root` write
  `NULL` for a non-member registration.
- **`reregister_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::reregister_instance`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2126-2173`): the SAME `instances`
  upsert, plus, INSIDE THE SAME TRANSACTION, an `INSERT ... ON
  CONFLICT(instance_id) DO UPDATE SET kinds, state` into `workers` — but ONLY
  when `reg.worker_snapshot()` (taken once, before the transaction) is
  `Some` (`crates/jammi-db/src/catalog/jobs_repo.rs:2155-2166`, the `if let
  Some(w) = worker { ... }` guard). This is the ONLY verb that re-creates a
  pruned row; `touch_instance`
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::touch_instance`, a
  pure `UPDATE ... WHERE instance_id = $2`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2177-2193`) can never resurrect
  one.
- **`fresh_instance(instance_id, lease)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fresh_instance`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2202-2230`): `true` iff a row is
  present AND NOT `stale_before_clause("last_seen_at", kind, margin, ...)`,
  where `margin = instance_liveness_margin(lease)` (`super::lease`, §5). No
  tenant predicate — `instances` carries no tenant column.
- **`peer_addr_of(instance_id, lease) -> Result<Option<PeerAddr>>`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::peer_addr_of`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2247-2287`): `SELECT peer_addr
  FROM instances WHERE instance_id = $N AND peer_addr IS NOT NULL AND NOT
  ({stale})` (`crates/jammi-db/src/catalog/jobs_repo.rs:2269-2272`); `None`
  when the row is absent, stale, or `peer_addr` is NULL, ALL collapsed the
  same way. A stored `peer_addr` that fails `PeerAddr::parse` is a typed
  `JammiError::Catalog`, never silently mapped to `None`. No kind/root/self
  filter: the ONE by-id resolution surface.
- **`list_gang_members(GangListing)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::list_gang_members`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2310-2381`): SQL is `SELECT
  i.instance_id, i.peer_addr, i.result_root, w.kinds, w.state FROM instances
  i JOIN workers w ON w.instance_id = i.instance_id WHERE i.peer_addr IS NOT
  NULL AND i.result_root IS NOT NULL AND NOT ({stale})` — an INNER join (a
  member must have a `workers` row), freshness/NULL-ness pushed into SQL.
  The Rust filter loop excludes, in order: the caller itself
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2351-2353`); `state !=
  WorkerState::Claiming.as_db_str()`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2354-2356`); `kinds.split(',').
  map(str::trim).any(|t| t == listing.kind)` false
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2357-2364`, a WHOLE trimmed
  token, never a substring); `result_root.as_bytes() !=
  listing.member_root.as_str().as_bytes()`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2365-2367`, a Rust BYTE
  comparison, never SQL `=`, never a re-interpreted value — this is the
  round-3 predicate). A surviving row's `peer_addr` is parsed and pushed as a
  `GangMember`; the final list is `members.sort_by(|a, b|
  a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()))`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2379`) — Rust byte-order sort,
  never a SQL `ORDER BY`.
- **`upsert_worker`**, **`set_worker_state`**, and **`delete_worker`** are
  unchanged in SQL shape from base; their callers are at §4. The sole
  production producer, `JobWorker::run_until`/`EmbeddedWorker::begin_drain`
  (`crates/jammi-ai/src/fine_tune/worker.rs:822,827,851`), always calls
  `self.kinds.join(",")`.
- **`prune_instances(stale_after)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::prune_instances`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2540`): `DELETE FROM instances
  WHERE {stale}` — its own `workers` row cascades (`ON DELETE CASCADE`,
  `crates/jammi-db/src/catalog/schema.rs:1030`).

## 4. The claim-loop's ownership of the worker half

`crates/jammi-ai/src/fine_tune/worker.rs`:

- **`JobWorker::run_until`**: sets the registration's worker cell to
  `Some(WorkerFacts { kinds: self.kinds.join(","), state:
  WorkerState::Warming })` (`crates/jammi-ai/src/fine_tune/worker.rs:821-824`)
  BEFORE issuing the first `upsert_worker`
  (`crates/jammi-ai/src/fine_tune/worker.rs:827`) — so a keeper reregister
  racing this very first upsert can never observe an empty cell. After the
  worker gate opens, the SAME cell-before-row order flips the cell's
  `state` to `Claiming`
  (`crates/jammi-ai/src/fine_tune/worker.rs:849-852`, `state:
  WorkerState::Claiming` at `:852` — round-3 citation correction: `:851` is
  the sibling `kinds: self.kinds.join(",")` line, not the state) before
  calling `set_worker_state`.
- **`EmbeddedWorker`** holds `registration: Arc<InstanceRegistration>`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2687`), cloned from
  `session.instance_registration()` at spawn
  (`crates/jammi-ai/src/fine_tune/worker.rs:2738`) — the SAME `Arc` the
  session's own `LeaseTarget::Instance` hold renews (confirmed:
  `crates/jammi-ai/src/session.rs:370` holds
  `LeaseTarget::Instance(Arc::clone(&registration))` from the SAME
  `registration` variable stored as `instance_registration: registration` at
  `crates/jammi-ai/src/session.rs:389`). `begin_drain` preserves the cell's
  `kinds`, flips only `state` to `Draining`, THEN calls `set_worker_state`.
  `stop_and_join` clears the cell
  (`crates/jammi-ai/src/fine_tune/worker.rs:2850`) BEFORE
  `self.catalog.delete_worker(&self.instance_id)`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2851`).
- **`InferenceSession::instance_registration()`**
  (`crates/jammi-ai/src/session.rs::InferenceSession::instance_registration`,
  `crates/jammi-ai/src/session.rs:472`) is the `pub(crate)` getter both
  `EmbeddedWorker::spawn` and the lease keeper's hold construction read
  from — the SAME `Arc` throughout a session's lifetime.

## 5. The prune window and the keeper's reregister

`crates/jammi-db/src/catalog/lease.rs`:

- **`instance_liveness_margin(lease) -> Duration`**
  (`crates/jammi-db/src/catalog/lease.rs::instance_liveness_margin`,
  `crates/jammi-db/src/catalog/lease.rs:287`): `lease.saturating_mul(2)`.
- **`instance_prune_window(lease) -> Duration`**
  (`crates/jammi-db/src/catalog/lease.rs::instance_prune_window`,
  `crates/jammi-db/src/catalog/lease.rs:301`):
  `instance_liveness_margin(lease).saturating_add(lease)`, i.e. `3 × lease` —
  STRICTLY beyond the margin (the self-test
  `crates/jammi-db/src/catalog/lease.rs::tests::
  prune_window_is_strictly_beyond_the_liveness_margin`).
- **`InferenceSession::wrap_with`**'s prune call uses this named function,
  never a literal `saturating_mul(2)`/`(3)` at the call site.
- **`renew_all`'s `Instance` arm**
  (`crates/jammi-db/src/catalog/lease_keeper.rs::renew_all`):
  `touch_instance(&reg.instance_id)`
  (`crates/jammi-db/src/catalog/lease_keeper.rs:772`) → `Ok(true)` renews;
  `Ok(false)` (a missed touch) calls `catalog.reregister_instance(reg)`
  (`crates/jammi-db/src/catalog/lease_keeper.rs:780`) INSTEAD of flipping
  `lost`.
- **`LeaseTarget::Instance(Arc<InstanceRegistration>)`**
  (`crates/jammi-db/src/catalog/lease_keeper.rs::LeaseTarget::Instance`,
  `crates/jammi-db/src/catalog/lease_keeper.rs:67`) carries the `Arc`, not an
  owned copy.

---

## 6. History — rounds 1–3 (superseded; no construct citations)

This section narrates what existed at earlier commits, for audit continuity
only. It cites no `path::construct`/`path:line` token, since every
construct it describes was DELETED by commit 1 and a citation into a
deleted construct would fail this revision's own machine-check (§7).

**Round 1** built the pure-validate/materialize split: a `CanonicalRoot`
type, an `EffectiveRoot` enum (`Default`/`ResultRootFile`/`ResultRootCloud`),
and a `MembershipConfig::materialize` method that `create_dir_all`'d and
`fs::canonicalize`d a `file://` anchor, appending a `jammi_db` leaf lexically
for the default arm. **Round 2**'s closing audit BLOCKed round 1's anchor
check: it had bound the absoluteness/UTF-8 guards to a `StorageUrl`-reparsed
VIEW of `artifact_dir` rather than the literal `PathBuf` every other
consumer read, so a `file://`- or cloud-scheme-spelled `artifact_dir` could
pass absoluteness as a URL while remaining relative as a path, and a
cloud-spelled `artifact_dir` could take the no-leaf cloud arm and name a
root the store never actually rooted at. The round-2 fix rebound the check
to the literal `PathBuf` and kept the materialize/canonicalize step.
**Round 3**'s closing audit BLOCKed a THIRD time: the cloud arm folded
`gcs://` → `gs://` and `abfss://` → `azure://` through `Scheme`'s `Display`
impl while the result store rooted itself at the VERBATIM string, so two
different-but-aliased spellings of `[storage] result_root` produced ONE
canonical string while the store itself was rooted at the string BEFORE the
fold — a false-positive membership match on cloud roots the alias fold
silently introduced. Per `CONTRACT-U5b-1a.md` §9's pre-committed round-3
stop rule ("a third BLOCK on M5 excises canonicalization entirely"),
canonicalization is deleted rather than patched a fourth time — no fourth
round on this mechanism. The excised design (scheme aliasing, symlink
resolution, case/trailing-slash folding) is filed as its own unit,
**U5b-1a-A2** ("result-root identity across spellings"), in
`docs/plans/67-distributed-training/README.md`'s unit table; its spec is
this history section plus rounds 1–3 of `docs/rigor/feat_500-C-U5b-1a.jsonl`.
It is explicitly NOT a wave-3 precondition.

---

## 7. Properties (quantified, never a single-input claim)

**P-M3 (the membership predicate).** For every
`list_gang_members(GangListing { kind, self_instance, member_root, lease })`
call and every DB row order: a row is returned iff it is NOT
`self_instance`, has a `workers` row (INNER join) with `state == Claiming`,
`kinds` contains `kind` as a whole trimmed comma-split token (a kind that is
merely a SUBSTRING of a token, e.g. `fine_tune` vs. `graph_fine_tune`, is
excluded —
`crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_kind_that_is_only_a_substring_token`),
`peer_addr` and `result_root` are both non-NULL, `result_root` bytes equal
`member_root` bytes (a root divergent only by case or a trailing `/` is
excluded —
`crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_root_divergent_by_case_or_trailing_slash`),
and it is fresh under `instance_liveness_margin(lease)`; the surviving set is
sorted by `instance_id` byte order regardless of the underlying row order
(`crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order`).
`peer_addr_of` resolves a busy or other-kind fresh member (no kind/state
filter) and is `None` for a stale or NULL-`peer_addr` one. Both verbs return
the SAME answer under a scoped tenant binding and under none
(`crates/jammi-db/tests/it/gang_membership.rs::list_gang_members_is_identical_under_a_scoped_tenant_and_under_none`).

**P-M4 (recovery restores the WHOLE membership tuple).** For a process that
IS a gang member, force-deleting its `instances` row and running one real
lease-keeper pass makes `list_gang_members` return it again, with
`peer_addr`, `result_root`, `kinds`, and `state` byte-identical to before the
delete — proven at the catalog layer
(`crates/jammi-db/tests/it/gang_membership.rs::keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete`)
and at the session layer through the session's OWN keeper
(`crates/jammi-ai/tests/it/instance_identity.rs::a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass`).
The dual: a DRAINED worker is NOT resurrected as a member after a forced
delete
(`crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`).

**P-prune-window.** A member merely stale in `(instance_liveness_margin(lease),
instance_prune_window(lease)]` is NOT pruned; one strictly beyond the window
IS
(`crates/jammi-db/tests/it/gang_membership.rs::prune_window_does_not_prune_a_member_merely_stale_within_the_window`,
`crates/jammi-ai/tests/it/instance_identity.rs::a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep`).

**P-M5 (over the constructor, round-3 restatement).** `peer_advertise`
without `peer_bind` → typed error naming both keys
(`crates/jammi-db/src/config/tests.rs::from_config_peer_advertise_without_peer_bind_is_refused_naming_both_keys`,
`::load_from_peer_advertise_without_peer_bind_is_refused_naming_both_keys`,
`crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_without_peer_bind_fails_open_naming_both_keys`,
the last through the REAL `InferenceSession::new`, and asserting ZERO
`instances` rows on the failure path too); a non-UTF-8 `artifact_dir` →
refused naming the key by `resolved_result_root` itself
(`crates/jammi-db/src/config/tests.rs::from_config_refuses_a_non_utf8_artifact_dir_via_resolved_result_root`);
`result_root` UNSET → accepted,
`{artifact_dir}/jammi_db`; `result_root` SET (to ANY well-formed or
ill-formed string, including a `memory://` root, an uppercase scheme, or a
path that does not exist) → accepted VERBATIM — the round-3 design refuses
NOTHING about `result_root`'s content
(`crates/jammi-ai/tests/it/instance_identity.rs::missing_result_root_path_is_accepted_verbatim_and_session_open_succeeds`);
a library config (no `peer_advertise`) → NULLs, never an error
(`crates/jammi-db/src/config/tests.rs::from_config_without_peer_advertise_is_a_library_registration_with_nulls`,
`crates/jammi-ai/tests/it/instance_identity.rs::library_config_without_peer_advertise_writes_null_membership_columns`).

**P-X1 (verbatim identity, contract §9 round-3).** For EVERY arm — unset,
`file://`, `memory://`, `s3://`, `gcs://`, `gs://`, `abfss://`, `azure://`,
an uppercase scheme, a trailing `/` —
`InstanceRegistration::from_config`'s `member_root` equals
`resolved_result_root()` byte-for-byte
(`crates/jammi-db/src/config/tests.rs::from_config_member_root_is_resolved_result_root_verbatim_over_every_arm`),
and that same string is what a real session's `instances.result_root`
column carries AND what the result store is rooted at, through the real
`InferenceSession` construction, for the unset, `file://`, `memory://`, and
the `gcs://` ALIAS arms
(`crates/jammi-ai/tests/it/storage_root.rs::member_row_matches_resolved_root_when_unset`,
`::member_row_matches_resolved_root_for_file_scheme`,
`::member_row_matches_resolved_root_for_memory_scheme`,
`::member_row_matches_resolved_root_for_a_cloud_alias_scheme`).

**P-X2 (byte-equality predicate, the join-time counterpart of P-X1).**
`gcs://b/p` and `gs://b/p` are DIFFERENT `member_root` values — never folded
on the membership-check path
(`crates/jammi-db/src/config/tests.rs::from_config_never_aliases_gcs_and_gs_result_root_spellings`)
— and two real gang members rooted at each spelling are NOT gang members of
each other
(`crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_not_gang_members_of_each_other`).
Necessary-never-sufficient for shared storage stays unchanged (§ Scope).

**P-X3 (no interpretation on the membership path, the grep oracle).** See
§7's self-check: `grep -ci 'canonical\|materialize\|EffectiveRoot'` is `0`
in `instance.rs`, `config/mod.rs`, `config/tests.rs`, `gang_membership.rs`,
`instance_identity.rs`, `storage_root.rs`; every remaining hit in
`jobs_repo.rs`/`schema.rs`/`session.rs` is a pre-existing, unrelated use
(TrainingSet materialization, a canonical UUID/timestamp spelling, a
`canonical constructor` doc phrase) — named individually in commit 1's own
body.

**P-X4 (the store and the row are one string, stated once; hardened this
revision — see the header note).** `MemberRoot` is constructible in
production ONLY from `MemberRoot::resolved`
(`crates/jammi-db/src/catalog/instance.rs::MemberRoot::resolved`, §1),
which `InstanceRegistration::from_config` (§2) is the only caller of, and
which itself calls `resolved_result_root` — never a caller-supplied
string. `MemberRoot::new`, the arbitrary-string wrapper, is compiled ONLY
under `feature = "test-hooks"` (§1): a real `cfg` gate, so it does not link
into a production build at all, unlike `#[doc(hidden)]` (which hides a
symbol from rendered docs but not from the compiler). This is the
ENFORCING half; the ENUMERATING half is a new, unconditional (no
`test-hooks` needed — a pure static sweep of the tree) oracle,
`crates/jammi-db/tests/it/member_root_constructor.rs::
member_root_new_has_no_production_caller`, which walks every
`crates/<name>/src/**/*.rs` file in the whole workspace (14 crates with a
`src/` directory, 429 `.rs` files at commit 1) and fails on the first
literal occurrence of `MemberRoot::new(` — so a future in-tree production
caller (even one that turns `test-hooks` on for an unrelated reason) is a
named, located finding, not a silent regression. Falsified before being
committed: a throwaway call injected into `lease.rs` made the oracle fail,
naming that file and line; reverted before commit 1.

**P-X5 (this contract).** History (§6) is kept, corrected to the RIGHT
constructs (the F1 narrative, §2); the citation round-3 items land (worker.rs
`:851→:852`, §4; the test name `peer_addr_parse_keeps_ipv6_host_intact`, §1);
every `path::construct`/`path:line` token in §1–§5, §7 is machine-checked
against commit 1's tree (§7's self-check).

**P-X6 (the excised design is filed, not lost).** See §6: unit U5b-1a-A2 is
in `docs/plans/67-distributed-training/README.md`'s unit table, NOT a
wave-3 precondition.

---

## 8. Oracles, by name, and what each EXCLUDES

| Oracle | Asserts | Excludes |
|---|---|---|
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_the_caller_itself` | The self row is never returned even though it is otherwise a valid member | Does not test a self row that is ALSO stale/wrong-kind |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_stale_member` | A row past the liveness margin is excluded | Does not probe the exact margin boundary — `gang_instance_freshness.rs`'s job |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_draining_worker` / `::list_excludes_a_warming_worker` | `state != Claiming` excludes, for BOTH non-claiming states independently | Neither combines a non-claiming state with a second false conjunct |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_kind_that_is_only_a_substring_token` | `fine_tune` never matches a `graph_fine_tune` token | Does not test the reverse substring direction |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_root_divergent_by_case_or_trailing_slash` | A byte-divergent root (case, trailing `/`) is excluded — byte-exact comparison | Does not test a symlink-equivalent path (that IS the excised design, filed at U5b-1a-A2) |
| `crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_not_gang_members_of_each_other` | P-X2: two real gang members rooted at aliased cloud spellings are not each other's members, in either listing direction | Does not test the `abfss://`/`azure://` pair separately (same alias-fold mechanism a config-level test already isolates) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_null_peer_addr` | `peer_addr IS NULL` excludes even with a fresh, correctly-kinded, claiming, root-matching row otherwise | Does not test `result_root IS NULL` in the same row (next oracle isolates it) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_member_with_peer_addr_set_but_result_root_null` | The asymmetric NULL case (migration 035 has no paired CHECK) is still excluded | Does not test the reverse (symmetric SQL predicate stands for both) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_an_instance_with_no_workers_row` | The INNER join excludes an `instances` row with no `workers` row at all | Does not test a `workers` row for a DIFFERENT instance_id colliding on kind |
| `crates/jammi-db/tests/it/gang_membership.rs::list_includes_a_fresh_multi_kind_claiming_worker` | The positive control: every conjunct held DOES return the member | Does not vary which of the two-or-more kinds is queried |
| `crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order` | Byte-order sort holds even off a descending-insertion-order raw SELECT, with a vacuity control | Does not test the postgres planner's own default order separately |
| `crates/jammi-db/tests/it/gang_membership.rs::list_gang_members_is_identical_under_a_scoped_tenant_and_under_none` | Tenant-independence, mirroring `get_job_for_rank`'s own oracle | Does not test under `with_admin_scope` (no admin-scope predicate exists on this path) |
| `crates/jammi-db/tests/it/gang_membership.rs::peer_addr_of_resolves_a_busy_or_other_kind_fresh_member` / three `::peer_addr_of_is_none_for_*` | No kind/state filter; three independent `None`-producing causes | Does not distinguish the three by return shape — all `None` |
| `crates/jammi-db/tests/it/gang_membership.rs::keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete` | P-M4 at the catalog layer, `peer_addr_of` too | Does not exercise the session-construction path (next file does) |
| `crates/jammi-db/tests/it/gang_membership.rs::prune_window_does_not_prune_a_member_merely_stale_within_the_window` | Both boundary directions in one test | Does not test EXACTLY at the window boundary |
| `crates/jammi-db/tests/it/member_root_constructor.rs::member_root_new_has_no_production_caller` | P-X4 (hardened, this revision): no `crates/<name>/src/**/*.rs` file in the whole workspace contains the literal call form `MemberRoot::new(` — a sanity floor on the walk itself (≥10 crates with a `src/` dir, ≥300 `.rs` files scanned) guards against a broken walk passing vacuously | Does not cover `benches/`/`examples/` (neither exists in this workspace today) or a call spelled through a type alias/re-export |
| `crates/jammi-db/src/config/tests.rs::from_config_member_root_is_resolved_result_root_verbatim_over_every_arm` | P-X1: the pure constructor's `member_root` == `resolved_result_root()` verbatim over the whole arm list, plus the "/" alone arm | Does not touch the filesystem — a real-session counterpart is `storage_root.rs` |
| `crates/jammi-db/src/config/tests.rs::from_config_never_aliases_gcs_and_gs_result_root_spellings` | P-X2 at the pure-function layer | Does not test the join predicate (gang_membership.rs's alias test does) |
| `crates/jammi-db/src/config/tests.rs::from_config_peer_advertise_without_peer_bind_is_refused_naming_both_keys` / `load_from_...` | P-M5's one remaining refusal arm, at both call sites | Does not test a malformed address string separately (`PeerAddr::parse`'s own unit tests cover that) |
| `crates/jammi-db/src/config/tests.rs::from_config_without_peer_advertise_is_a_library_registration_with_nulls` | A library config produces NULL/NULL, never an error | Does not test a session that later sets `peer_advertise` (no runtime mutation path exists) |
| `crates/jammi-db/src/config/tests.rs::from_config_refuses_a_non_utf8_artifact_dir_via_resolved_result_root` | The ONE surviving refusal on the whole path, propagated from `resolved_result_root` | Does not test a non-UTF-8 explicit `result_root` (an already-`String` config field — non-UTF-8 there is not representable in TOML/env at all) |
| `crates/jammi-ai/tests/it/storage_root.rs::member_row_matches_resolved_root_when_unset` / `_for_file_scheme` / `_for_memory_scheme` / `_for_a_cloud_alias_scheme` | P-X1 through a REAL session: the row, `resolved_result_root()`, and the store's own root all agree, for four arms including one alias | The `abfss://`/`azure://` alias direction is covered at the config layer only (network-free); a real-session probe of an unconfigured cloud scheme risks a live network attempt for the other three schemes, so exactly one (`gcs://`) is chosen |
| `crates/jammi-ai/tests/it/instance_identity.rs::library_config_without_peer_advertise_writes_null_membership_columns` | A real session with no `peer_advertise` writes NULL/NULL | — |
| `crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_set_result_root_unset_produces_a_nonnull_row_via_open` / `_set_produces_a_nonnull_row_via_open_with_placement` | The row's `result_root` equals `resolved_result_root()`'s own value, through `open` and `open_with_placement` respectively | Does not test a cloud-scheme `result_root` through this real-construction path (`storage_root.rs` covers that) |
| `crates/jammi-ai/tests/it/instance_identity.rs::peer_advertise_without_peer_bind_fails_open_naming_both_keys` | The typed-error path through `InferenceSession::new`, AND zero `instances` rows written | Does not test the same failure through `open`/`open_with_placement` (all three funnel through `wrap_with`) |
| `crates/jammi-ai/tests/it/instance_identity.rs::missing_result_root_path_is_accepted_verbatim_and_session_open_succeeds` | A not-yet-existing `result_root` path is accepted VERBATIM — the membership check itself never touches the filesystem (the path IS later created, but by the result store's own local-root `create_dir_all`, a separate mechanism) | Does not assert the anchor is never created at all (that claim is false post-excision; the store still creates a local `file://` root) |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass` | P-M4 through the session's OWN keeper, with a real `EmbeddedWorker` | Does not test a session with NO worker spawned (next oracle) |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete` | `instances` reregisters, `workers` does not | Does not test a re-spawn of a NEW worker after the drain |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep` | The construction-sweep prune call uses `instance_prune_window`, not the old literal, through a REAL second session's boot | Does not test the symmetric past-window-is-pruned case through this real path |
| `crates/jammi-db/tests/it/migrations.rs::migration_035_is_ordered_after_034_and_adds_instances_peer_addr_result_root` | Ordering; both columns nullable on both dialects; a both-NULL row and a both-set row are both valid inserts | Does not test a row with exactly one of the two set (gang_membership.rs's own oracle covers that shape's listing behavior) |
| `crates/jammi-db/tests/it/migrations.rs::migration_029_copies_training_jobs_rows_into_jobs_as_queued` | The fourth K5 pin site: `035` is on the ledger's manufactured-pre-029 DELETE list | Proves the pin site's absence would be RED; does not itself remove the entry to observe it |
| `crates/jammi-db/src/catalog/lease.rs::tests::instance_liveness_margin_is_twice_the_lease` / `::prune_window_is_strictly_beyond_the_liveness_margin` | The `2×`/`3×` factors and the `Duration::MAX` saturating-arithmetic edge | Does not test a lease of zero (validated positive upstream) |
| `crates/jammi-db/src/catalog/instance.rs::tests::registration_worker_half_starts_unpopulated_and_is_settable` | The cell's own set/clear/snapshot round-trip in isolation | Does not test concurrent set/snapshot from two threads |
| `crates/jammi-db/src/catalog/instance.rs::tests::peer_addr_parses_host_port` / `::peer_addr_parse_keeps_ipv6_host_intact` / `::peer_addr_refuses_no_colon` / `::_refuses_empty_host` / `::_refuses_zero_port` / `::_refuses_non_numeric_port` / `::_refuses_out_of_range_port` | Every `PeerAddr::parse` edge, including the IPv6-colon-preservation shape | Does not test a hostname requiring DNS resolution |

---

## 9. Mutations executed / implied by the oracle shapes above

- **The self-exclusion / state-exclusion / kind-token-match / root-byte-compare
  conjuncts.** Each has its own isolated oracle (§8 table) holding every
  other conjunct at its satisfied value and flipping exactly one — deleting
  any single `continue` arm in `list_gang_members`'s Rust filter loop
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2351-2367`) flips exactly the
  corresponding test from a correct exclusion to a false inclusion.
- **The `sort_by` call.** Deleting
  `crates/jammi-db/src/catalog/jobs_repo.rs:2379` is exactly what
  `crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order`'s
  raw-order control is built to catch.
- **`instance_prune_window` vs. the old literal.** Reverting
  `crates/jammi-ai/src/session.rs`'s prune call to
  `lease_intervals.lease().saturating_mul(2)` is exactly what
  `crates/jammi-ai/tests/it/instance_identity.rs::a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep`
  is built to catch.
- **`reregister_instance`'s worker-half conditional.** Removing the `if let
  Some(w) = worker { ... }` guard at
  `crates/jammi-db/src/catalog/jobs_repo.rs:2155-2166` would flip
  `crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`'s
  expected "no workers row after recovery" to a false resurrection.
- **The verbatim-identity property (round-3, replaces the round-2 mutation
  about the literal-`PathBuf`-vs-URL-reparse check, since that check no
  longer exists).** Re-introducing ANY scheme fold in
  `InstanceRegistration::from_config` (`crates/jammi-db/src/catalog/
  instance.rs:239-255`) — e.g. lower-casing or alias-folding the
  `member_root` string before wrapping it — is exactly what
  `crates/jammi-db/src/config/tests.rs::from_config_never_aliases_gcs_and_gs_result_root_spellings`
  and `crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_not_gang_members_of_each_other`
  are built to catch: both fixtures assert the two spellings stay distinct
  strings and distinct members.

---

## 10. What is NOT in this unit

- **Any caller of `peer_addr_of`/`list_gang_members`** — both are
  unreachable from any public RPC at this head, vacuously; U5b-1b-ii owes the
  server-side unreachability oracle when it adds the first caller (restated
  at `docs/guide/src/security.md`'s I-GANG bullet).
- **The attestation VERIFY** that gives root-equality sufficiency — U5a-1's
  whole-artifact sidecar and U5b-0's/U5b-1b-i's per-partition inventory are
  separate mechanisms; this unit states the necessary-never-sufficient
  boundary but builds no new verify path.
- **`RendezvousPlacement`** and any query-time consumption of the ring
  (DIST's own unit, downstream of this substrate).
- **A capability-scoped ring** — a replica that sets `peer_advertise` also
  joins the retrieval ring by construction; scoping the two independently is
  a later plan's follow-on.
- **Spelling identity across schemes/symlinks/case** — filed as unit
  U5b-1a-A2 (§6), NOT a wave-3 precondition.

---

## 11. Report

`impossibility_claims`:
- "no caller other than `InstanceRegistration::from_config` builds a value
  `Catalog::upsert_instance`/`Catalog::reregister_instance` accept" — not
  independently enumerated by a code-scanning oracle in this unit; the claim
  rests on the two verbs' own signatures (`&InstanceRegistration`) rather
  than an executed grep-shaped enumeration — `uncovered`.
- "a stored `peer_addr` that fails to parse is never silently treated as
  absent" — executed check: both `peer_addr_of` and `list_gang_members`
  return a typed `JammiError::Catalog` on a parse failure — verified by
  direct read of both call sites; no test manufactures a corrupted row —
  `uncovered` as a standing tripwire.
- "the `,` kind-encoding convention is honored by every real writer" —
  executed check: `crates/jammi-ai/src/fine_tune/worker.rs:822,827,851` are
  the ONLY three call sites of `self.kinds.join(",")` in this crate
  (confirmed by direct read, not a code-scanning oracle) — `uncovered`.
- (round-3) "canonicalization ever made two spellings of one location
  provably identical to a third party" — this claim was never true (rounds
  1–2 canonicalized only the LOCAL `file://` case; a cloud root was already
  passed through with, at best, a scheme-alias fold that round 3 found
  itself unsound); the round-3 design does not attempt it — the necessity
  boundary (§ Scope) always covered this gap, now honestly with no
  machinery pretending otherwise.

`citations_reanchored`: every citation in §1–§5, §7–§9 of this revision was
re-derived directly against the tree of `686d4aedfefb3693ee1b798a8749f63493df1acc`
(commit 1, re-anchored — see the header's P-X4 hardening note) in this
worktree by the writing agent; §6 (history) cites no construct at all, by
design (see §6's own header).

**Self-check (round-2 addendum requirement; re-run for the P-X4 hardening
re-anchor).** Before this revision was committed, every `path:line`/
`path:a-b` token AND every `path::construct` token in §1–§5, §7–§9 (§6
excluded, cites none) was extracted and checked against the tree of commit
1: a `path:line` token's cited line(s) were printed and eyeballed against
the identifier/claim the surrounding sentence names; a `path::construct`
token's LAST path segment (after the final `::`) was grepped against the
named file, confirming the construct's name still occurs there.

```
python3 /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/12f161bf-7977-4318-a34d-d2eec24a0620/scratchpad/logs/db-selfcheck-px4.py
```

(a scratch script, never committed). **Result: 86 `path::construct` tokens
checked, 86 resolved (their last segment occurs in the named file); 55
unique `path:line`/`path:a-b` tokens checked, 55 resolved (the named file
has at least that many lines) — both counts 100%, zero unresolved.**

---

## 12. Gates

`cargo fmt --all --check`; `cargo clippy -p jammi-db -p jammi-ai -p
jammi-server --all-targets -- -D warnings` (with AND without
`--features test-hooks` on `jammi-db`, since `MemberRoot::new` and the
`gang_membership.rs` module it is built through are `cfg`-gated on it —
`jammi-ai`'s own test targets already turn `jammi-db/test-hooks` on
unconditionally via its self dev-dependency, so only the `jammi-db`-alone
invocation needs the explicit flag both ways); `cargo test -p jammi-db -p
jammi-ai` (the plain, `test-hooks`-off run: `gang_membership.rs` is
`#[cfg(feature = "test-hooks")]`-gated at `tests/it/main.rs` and does not
compile in this pass, same as `materialization_crash_recovery.rs`/
`mutable_crash_recovery.rs`; `member_root_constructor.rs`'s enumerating
oracle is unconditional and DOES run here) AND `cargo test -p jammi-db
--features test-hooks --test it` (the lane `gang_membership.rs` actually
runs in — `.github/workflows/ci.yml`'s "test-hooks lane" step, on every
PR) and `cargo test --workspace` (a server-startup-adjacent config change —
other crates' test harnesses spawn the binary with their own env); the
live-postgres lane (`--features live-postgres-tests`, `JAMMI_TEST_PG_URL`
set) for every `test_case`-parameterized test in
`crates/jammi-db/tests/it/gang_membership.rs` and
`crates/jammi-db/tests/it/migrations.rs`'s 035 oracle;
`python3 ci/scripts/check_doc_parity.py`; `python3 ci/scripts/perf/
check_citations.py`; `python3 ci/scripts/check_no_consumer_names.py`;
`python3 ci/scripts/check_swarm_bijection.py`; `RUSTDOCFLAGS="-D warnings"
cargo doc --no-deps -p jammi-db -p jammi-ai`.
