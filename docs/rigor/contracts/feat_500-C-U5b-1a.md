# CONTRACT — feat/500-C-U5b-1a: gang-membership substrate

**Contract of record.** slug: `feat_500-C-U5b-1a` · branch `feat/500-C-U5b-1a`
at `13663cc178ab0f74a0f2811ff9aa725ba4a198b4` (the code commit, tagged
throughout this revision as "commit 1") · this file is the committed
mechanism contract `ci/scripts/check_rigor_record.py` requires under
`docs/rigor/contracts/**` (its check 3) before this unit's rigor record at
`docs/rigor/feat_500-C-U5b-1a.jsonl` (the lead's export, committed LAST —
after every closer whose rows it carries, as the final commit before the
PR) satisfies check 1/2. Source design contract: the lead's
`CONTRACT-U5b-1a.md` v4 (scratchpad-only, never a repo path — it is not
cited with a `path:line` token anywhere in this document, since it is not a
tracked file at HEAD).

**Round-5 doc-fold re-anchor (this revision).** "Commit 1" moves from
`686d4aedfefb3693ee1b798a8749f63493df1acc` to
`3b7c89d4125862ef0c7f9c5073dc80203b81f2cf`: the round-4 closing audit (§11)
found the round-3 excision's documentation fold incomplete (no mechanism
change), and this commit closes it — `WorkerFacts::kinds`'s and
`upsert_worker`'s doc comments drop the "otherwise producer-encoded" hatch,
`advertising_config`'s doc (`config/tests.rs`) now describes the shape its
body builds, and `gang_membership.rs` gains the corrupted-`peer_addr`
typed-error oracle on both read verbs. These are doc-comment/test-only
insertions in `instance.rs` (+3 lines, before
`MembershipConfig`/`InstanceRegistration`/`GangListing`/`GangMember`),
`jobs_repo.rs` (+2 lines, before `prune_instances`), and `config/tests.rs`
(+2 lines) — every
`path:line`/`path:a-b` citation into those three files below was
RE-DERIVED against THIS commit's tree (self-check, §11); citations by
`path::construct` are unaffected (no construct was renamed or moved).

**Round-6 doc-fold re-anchor (this revision, FINAL).** "Commit 1" moves from
`3b7c89d4125862ef0c7f9c5073dc80203b81f2cf` to
`13663cc178ab0f74a0f2811ff9aa725ba4a198b4`: round 5's pre-committed rule
fired (§11/§12) — `result_root` leaves `list_gang_members`'s admission
predicate entirely (`GangListing` loses its root field; the Rust filter
loses its fourth conjunct; `GangCandidateRow` loses its `result_root`
field), `PeerAddr::parse` gains the unbracketed-IPv6 refusal (P-Y4), and
`JobWorker::run_until`'s FIRST `upsert_worker` call reverses to
cell-after-row (P-Y4), gated by a NEW test-only module,
`crates/jammi-db/src/catalog/worker_test_hooks.rs`. Every
`path:line`/`path:a-b` citation in §1–§5, §7–§9 below was RE-DERIVED
against THIS commit's tree (self-check, §12); citations by
`path::construct` are unaffected except where the round-5 verdicts (§12)
name a moved or deleted construct explicitly.

**P-X4 hardening (round-3→4 revision, re-anchored "commit 1" then).** The prior commit
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
  `crates/jammi-db/src/catalog/instance.rs:43`): the only way to build one is
  `PeerAddr::parse` (`crates/jammi-db/src/catalog/instance.rs::PeerAddr::
  parse`, `crates/jammi-db/src/catalog/instance.rs:57`), which
  `rsplit_once(':')`s the input so a BRACKETED IPv6 literal's own colons stay
  inside the host segment and only the LAST colon splits host from port,
  requires a non-empty host, REFUSES an UNBRACKETED IPv6 host (P-Y4, §12 —
  `host.contains(':') && !(starts_with('[') && ends_with(']'))`), and
  requires a port that parses as `u16` and is non-zero. `as_str()`
  (`crates/jammi-db/src/catalog/instance.rs:86`) and `Display`
  (`crates/jammi-db/src/catalog/instance.rs:91`) both return the validated
  wire form. This is the SAME type `crates/jammi-db/src/index/peer.rs`
  re-exports (per this module's own doc comment) — the peer listener and the
  gang listener dial the identical address type, so they can never drift
  into two.
- **`MemberRoot`** (`crates/jammi-db/src/catalog/instance.rs::MemberRoot`,
  declared `crates/jammi-db/src/catalog/instance.rs:123`) wraps a string
  VERBATIM — no canonicalization, no interpretation, and, as of this
  revision, **not consulted by `list_gang_members` at all** (P-Y1, §12).
  `Self::resolved` (`crates/jammi-db/src/catalog/instance.rs::MemberRoot::
  resolved`, `crates/jammi-db/src/catalog/instance.rs:132`) is the ONE
  production constructor — it calls `resolved_result_root` itself, so a
  `MemberRoot` can never carry a string the resolver did not produce;
  `InstanceRegistration::from_config` (§2) is its only caller, and the value
  is still written to `instances.result_root` on every member row (P-Y2) —
  carried for `docs/plans/67-distributed-training/README.md` unit
  U5b-1a-A2, which now owns BOTH root identity across spellings and any
  membership predicate built on it. `Self::new`
  (`crates/jammi-db/src/catalog/instance.rs::MemberRoot::new`,
  `crates/jammi-db/src/catalog/instance.rs:145`) wraps an arbitrary string
  with no resolver call at all and is compiled ONLY under
  `feature = "test-hooks"` — a real `cfg` gate, not `#[doc(hidden)]` — so it
  cannot link into a production build at all; fixtures/tests build a
  `MemberRoot` through it directly
  (`crates/jammi-db/tests/it/gang_membership.rs`,
  `crates/jammi-ai/tests/it/instance_identity.rs`). **Correction (this
  revision): `crates/jammi-db/src/config/tests.rs` never builds a
  `MemberRoot` through `new` at all** — it reads `member_root` off an
  `InstanceRegistration` `from_config` already produced (§7's P-X1 oracle);
  the round-4/5 text above listing it as a `MemberRoot::new` call site was
  wrong (the §1 fixture-list error this revision corrects, per the lead's
  brief).
- **`WorkerFacts`** (`crates/jammi-db/src/catalog/instance.rs::WorkerFacts`,
  declared `crates/jammi-db/src/catalog/instance.rs:172`) is the claim-loop
  half of a registration: `kinds: String` (the comma-joined kind set) and
  `state: WorkerState`. Its doc states its SOLE owner is
  `JobWorker`/`EmbeddedWorker` (`crates/jammi-ai/src/fine_tune/worker.rs`),
  confirmed at §4 below, including the P-Y4 reversal on the FIRST
  `upsert_worker` call.
- **`InstanceRegistration`**
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration`, declared
  `crates/jammi-db/src/catalog/instance.rs:196`): `instance_id`, `label`,
  `host`, `peer_addr: Option<PeerAddr>`, `member_root: Option<MemberRoot>`,
  and `worker: Mutex<Option<WorkerFacts>>` — the claim-loop cell, mutated in
  place by its single owner rather than requiring a whole new registration
  per state change. `InstanceRegistration::new`
  (`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::new`,
  `crates/jammi-db/src/catalog/instance.rs:214`) is the plain constructor
  with no validation, used by `from_config` itself and by fixtures/tests
  (`crates/jammi-db/tests/it/gang_membership.rs`'s `seed_member` helper and
  direct-construction call sites, and
  `crates/jammi-ai/tests/it/instance_identity.rs`'s foreign-row fixture).
  `set_worker`/`worker_snapshot`
  (`crates/jammi-db/src/catalog/instance.rs:233`, `:238`) are the cell's
  read/write pair (mutex-poisoning tolerant via
  `unwrap_or_else(|p| p.into_inner())`).
- **`GangListing<'a>`** (`crates/jammi-db/src/catalog/instance.rs::
  GangListing`, `crates/jammi-db/src/catalog/instance.rs:325`) carries **no
  root field** (P-Y1, §12 — `{ kind, self_instance, lease }` only) and
  **`GangMember`** (`crates/jammi-db/src/catalog/instance.rs::GangMember`,
  `crates/jammi-db/src/catalog/instance.rs:341`) are `list_gang_members`'s
  request/response shapes (§3).

## 2. The membership check: `MembershipConfig::validate`, `InstanceRegistration::from_config`, `resolved_result_root`

**The round-3 design, in one paragraph (round-6 line re-derivation: this
revision).** `MembershipConfig::validate`
(`crates/jammi-db/src/catalog/instance.rs::MembershipConfig::validate`,
`crates/jammi-db/src/catalog/instance.rs:304-315`) is PURE and checks
EXACTLY two things: `[server] peer_advertise` parses as a `PeerAddr`
(`:313`), and `[server] peer_bind` is set too, else a typed
`JammiError::Config` naming BOTH keys (`:308-311`). It inspects
`result_root`/`artifact_dir` NOT AT ALL. `InstanceRegistration::from_config`
(`crates/jammi-db/src/catalog/instance.rs::InstanceRegistration::from_config`,
`crates/jammi-db/src/catalog/instance.rs:265-282`) runs `validate`; when
membership applies, it sets `member_root` to
`MemberRoot::resolved(config)?` (`:274`), which is itself
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
NOT the home for either check: it has no access to
`artifact_dir`/`storage.result_root`, which `resolved_result_root` needs.

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
  `crates/jammi-db/src/catalog/jobs_repo.rs:2081-2113`): one `INSERT ...
  VALUES (...) ON CONFLICT(instance_id) DO UPDATE SET label, host,
  peer_addr, result_root, last_seen_at`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2092-2098`) — `started_at` is
  stamped only on the initial insert. `reg.peer_addr`/`reg.member_root` write
  `NULL` for a non-member registration; `result_root` is still written here,
  verbatim (P-Y2), even though no read verb consults it (below).
- **`reregister_instance(&InstanceRegistration)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::reregister_instance`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2125-2172`): the SAME `instances`
  upsert, plus, INSIDE THE SAME TRANSACTION, an `INSERT ... ON
  CONFLICT(instance_id) DO UPDATE SET kinds, state` into `workers` — but ONLY
  when `reg.worker_snapshot()` (taken once, before the transaction) is
  `Some` (`crates/jammi-db/src/catalog/jobs_repo.rs:2154-2166`, the `if let
  Some(w) = worker { ... }` guard). This is the ONLY verb that re-creates a
  pruned row; `touch_instance`
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::touch_instance`, a
  pure `UPDATE ... WHERE instance_id = $2`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2176-2192`) can never resurrect
  one.
- **`fresh_instance(instance_id, lease)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::fresh_instance`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2201-2229`): `true` iff a row is
  present AND NOT `stale_before_clause("last_seen_at", kind, margin, ...)`,
  where `margin = instance_liveness_margin(lease)` (`super::lease`, §5). No
  tenant predicate — `instances` carries no tenant column.
- **`peer_addr_of(instance_id, lease) -> Result<Option<PeerAddr>>`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::peer_addr_of`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2246-2286`): `SELECT peer_addr
  FROM instances WHERE instance_id = $N AND peer_addr IS NOT NULL AND NOT
  ({stale})` (`crates/jammi-db/src/catalog/jobs_repo.rs:2267-2271`); `None`
  when the row is absent, stale, or `peer_addr` is NULL, ALL collapsed the
  same way. A stored `peer_addr` that fails `PeerAddr::parse` is a typed
  `JammiError::Catalog`, never silently mapped to `None`. No kind/self
  filter (never had a root filter either): the ONE by-id resolution surface.
- **`list_gang_members(GangListing)`** (P-Y1, contract §12, the round-5
  excision — this revision's mechanism description supersedes the round-3
  text below the line)
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::list_gang_members`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2316-2381`): `GangListing` is
  `{ kind, self_instance, lease }` — **no root field at all**. SQL is
  `SELECT i.instance_id, i.peer_addr, w.kinds, w.state FROM instances i JOIN
  workers w ON w.instance_id = i.instance_id WHERE i.peer_addr IS NOT NULL
  AND NOT ({stale})`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2331-2336`) — an INNER join (a
  member must have a `workers` row), freshness/NULL-`peer_addr`-ness pushed
  into SQL; `result_root` is neither selected nor filtered on, anywhere in
  this SQL. The Rust filter loop excludes, in order: the caller itself
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2354-2356`); `state !=
  WorkerState::Claiming.as_db_str()`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2357-2359`); `kinds.split(',').
  map(str::trim).any(|t| t == listing.kind)` false
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2360-2367`, a WHOLE trimmed
  token, never a substring) — **and NOTHING else**: the round-3 predicate's
  fourth conjunct (`result_root.as_bytes() != listing.member_root...`) is
  DELETED, not merely unreachable — `GangCandidateRow` (`crates/jammi-db/src/catalog/jobs_repo.rs:566`)
  no longer even carries a `result_root` field. A surviving row's
  `peer_addr` is parsed and pushed as a `GangMember`; the final list is
  `members.sort_by(|a, b|
  a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()))`
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2379`) — Rust byte-order sort,
  never a SQL `ORDER BY`.
- **`upsert_worker`**, **`set_worker_state`**, and **`delete_worker`** are
  unchanged in SQL shape from base; their callers are at §4. The sole
  production producer, `JobWorker::run_until`/`EmbeddedWorker::begin_drain`
  (`crates/jammi-ai/src/fine_tune/worker.rs:829,836,866`), always calls
  `self.kinds.join(",")`. `upsert_worker`'s doc gains an `# Errors` section
  (this revision): under `feature = "test-hooks"`, a failure armed through
  `crates/jammi-db/src/catalog/worker_test_hooks.rs::
  arm_upsert_worker_failure` (a NEW module, mirroring `claim_test_hooks.rs`'s
  shape — a `Mutex<Vec<String>>` of armed instance ids, one-shot, consumed by
  `take_armed` at the top of `upsert_worker`) is returned with no write
  attempted — the ONLY way to manufacture "this process's first `workers`
  upsert fails" deterministically (P-Y4's oracle, §12).
- **`prune_instances(stale_after)`**
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::prune_instances`,
  `crates/jammi-db/src/catalog/jobs_repo.rs:2559`): `DELETE FROM instances
  WHERE {stale}` — its own `workers` row cascades (`ON DELETE CASCADE`,
  `crates/jammi-db/src/catalog/schema.rs:1030`).

## 4. The claim-loop's ownership of the worker half

`crates/jammi-ai/src/fine_tune/worker.rs`:

- **`JobWorker::run_until`** (P-Y4, contract §12, the round-5 advisory —
  this revision REVERSES the round-3/4 "cell before row" description for
  the FIRST `upsert_worker` call only): sets the registration's worker cell
  to `Some(WorkerFacts { kinds: self.kinds.join(","), state:
  WorkerState::Warming })`
  (`crates/jammi-ai/src/fine_tune/worker.rs:832-838`) only AFTER the FIRST
  `upsert_worker` call (`crates/jammi-ai/src/fine_tune/worker.rs:827-830`)
  returns `Ok(())` — an `Err` arm leaves the cell untouched (`None`)
  (`crates/jammi-ai/src/fine_tune/worker.rs:840-842`), so a keeper
  reregister racing a STILL-FAILING loop start never writes a `workers` row
  the real upsert never itself managed to write. After the worker gate
  opens, every LATER cell write — the `state: Claiming` transition
  (`crates/jammi-ai/src/fine_tune/worker.rs:863-868`, `state:
  WorkerState::Claiming` at `:867`) before calling `set_worker_state`, and
  every `begin_drain`/`delete_worker` call site below — is UNCHANGED:
  cell-before-row, since by that point the row already exists (the first
  upsert already succeeded, or none of these later call sites would ever
  run with a populated cell to transition from).
- **`EmbeddedWorker`** holds `registration: Arc<InstanceRegistration>`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2702`), cloned from
  `session.instance_registration()` at spawn
  (`crates/jammi-ai/src/fine_tune/worker.rs:2753`) — the SAME `Arc` the
  session's own `LeaseTarget::Instance` hold renews (confirmed:
  `crates/jammi-ai/src/session.rs:370` holds
  `LeaseTarget::Instance(Arc::clone(&registration))` from the SAME
  `registration` variable stored as `instance_registration: registration` at
  `crates/jammi-ai/src/session.rs:389`). `begin_drain` preserves the cell's
  `kinds`, flips only `state` to `Draining`, THEN calls `set_worker_state`.
  `stop_and_join` clears the cell
  (`crates/jammi-ai/src/fine_tune/worker.rs:2865`) BEFORE
  `self.catalog.delete_worker(&self.instance_id)`
  (`crates/jammi-ai/src/fine_tune/worker.rs:2866`).
- **`InferenceSession::instance_registration()`**
  (`crates/jammi-ai/src/session.rs::InferenceSession::instance_registration`,
  `crates/jammi-ai/src/session.rs:475`) is the `pub(crate)` getter both
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

**P-M3 (the membership predicate, NARROWED this revision — see P-Y1, §12).**
For every `list_gang_members(GangListing { kind, self_instance, lease })`
call (no root field) and every DB row order: a row is returned iff it is NOT
`self_instance`, has a `workers` row (INNER join) with `state == Claiming`,
`kinds` contains `kind` as a whole trimmed comma-split token (a kind that is
merely a SUBSTRING of a token, e.g. `fine_tune` vs. `graph_fine_tune`, is
excluded —
`crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_kind_that_is_only_a_substring_token`),
`peer_addr` is non-NULL, and it is fresh under
`instance_liveness_margin(lease)` — **and NOTHING else**: `result_root` is
NOT a conjunct (a NULL `result_root`, and a `result_root` byte-divergent by
case, trailing `/`, or scheme alias, all still ADMIT —
`crates/jammi-db/tests/it/gang_membership.rs::
list_includes_a_member_with_peer_addr_set_and_result_root_null_root_is_not_consulted`,
`::list_includes_members_despite_a_root_divergent_by_case_or_trailing_slash_root_is_not_consulted`,
`::gcs_and_gs_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`,
`::file_and_s3_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`);
the surviving set is sorted by `instance_id` byte order regardless of the
underlying row order
(`crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order`).
`peer_addr_of` resolves a busy or other-kind fresh member (no kind/state
filter, and never had a root filter either) and is `None` for a stale or
NULL-`peer_addr` one. Both verbs return the SAME answer under a scoped
tenant binding and under none
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

**P-X2 (byte-equality of the VALUE, superseded at the join layer by P-Y1
this revision).** `gcs://b/p` and `gs://b/p` are DIFFERENT `member_root`
values — never folded when the value is WRITTEN
(`crates/jammi-db/src/config/tests.rs::from_config_never_aliases_gcs_and_gs_result_root_spellings`).
This is a claim about the STRING only: as of the round-5 excision (P-Y1,
§12), `list_gang_members` does not compare `result_root` at all, so two
real gang members rooted at each spelling ARE gang members of each other
(the test asserting this is renamed to say so:
`crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`,
and the same property is proven again over an unrelated scheme pair,
`::file_and_s3_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`).
Necessary-never-sufficient for shared storage stays unchanged (§ Scope) —
it was never more than that even when byte-equality WAS a join-time
conjunct.

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
`src/` directory, 430 `.rs` files at commit 1) and fails on the first
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
| `crates/jammi-db/tests/it/gang_membership.rs::list_includes_members_despite_a_root_divergent_by_case_or_trailing_slash_root_is_not_consulted` (renamed, round-6) | P-Y1: a byte-divergent root (case, trailing `/`) does NOT exclude — `result_root` is not part of the predicate at all | Does not test a symlink-equivalent path (that identity question is U5b-1a-A2's, unbuilt) |
| `crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_gang_members_of_each_other_root_is_not_consulted` (renamed, round-6) | P-Y1/P-X2: two real gang members rooted at aliased cloud spellings ARE each other's members — one `list_gang_members` call, no root argument to even supply | Does not test the `abfss://`/`azure://` pair separately (the sibling `file://`/`s3://` oracle proves the property is not scheme-pair-specific) |
| `crates/jammi-db/tests/it/gang_membership.rs::file_and_s3_spelled_members_are_gang_members_of_each_other_root_is_not_consulted` (new, round-6) | The same P-Y1 property over an UNRELATED scheme pair that never aliases anywhere in the stack, ruling out "the two schemes alias at some other layer" as an alternative explanation | Does not test a `memory://` root in this pairing (covered singly elsewhere) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_a_null_peer_addr` | `peer_addr IS NULL` excludes even with a fresh, correctly-kinded, claiming row otherwise | Does not test `result_root IS NULL` in the same row (next oracle isolates it — and shows it does NOT exclude) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_includes_a_member_with_peer_addr_set_and_result_root_null_root_is_not_consulted` (renamed, round-6) | P-Y1: the asymmetric NULL case (migration 035 has no paired CHECK) does NOT exclude — `result_root` plays no part in admission, NULL or otherwise; `peer_addr_of` (no root predicate ever) still resolves it too | Does not test the reverse (symmetric SQL predicate never existed for `result_root` even before this round) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_excludes_an_instance_with_no_workers_row` | The INNER join excludes an `instances` row with no `workers` row at all | Does not test a `workers` row for a DIFFERENT instance_id colliding on kind |
| `crates/jammi-db/tests/it/gang_membership.rs::list_includes_a_fresh_multi_kind_claiming_worker` | The positive control: every conjunct held DOES return the member | Does not vary which of the two-or-more kinds is queried |
| `crates/jammi-db/tests/it/gang_membership.rs::list_is_sorted_by_instance_id_bytes_despite_descending_insertion_order` | Byte-order sort holds even off a descending-insertion-order raw SELECT, with a vacuity control | Does not test the postgres planner's own default order separately |
| `crates/jammi-db/tests/it/gang_membership.rs::list_gang_members_is_identical_under_a_scoped_tenant_and_under_none` | Tenant-independence, mirroring `get_job_for_rank`'s own oracle | Does not test under `with_admin_scope` (no admin-scope predicate exists on this path) |
| `crates/jammi-db/tests/it/gang_membership.rs::peer_addr_of_resolves_a_busy_or_other_kind_fresh_member` / three `::peer_addr_of_is_none_for_*` | No kind/state filter; three independent `None`-producing causes | Does not distinguish the three by return shape — all `None` |
| `crates/jammi-db/tests/it/gang_membership.rs::peer_addr_of_returns_the_typed_error_for_a_corrupted_peer_addr` (round-5) | A `peer_addr` corrupted out-of-band (never through `PeerAddr::parse`) surfaces the typed `JammiError::Catalog` naming the instance, never a panic and never a silent `None` | Does not test a corrupted `result_root` column (not parsed, so not applicable) |
| `crates/jammi-db/tests/it/gang_membership.rs::list_gang_members_returns_the_typed_error_for_a_corrupted_peer_addr` (round-5) | The `list_gang_members` sibling: a matching, fresh, claiming candidate with a corrupted `peer_addr` surfaces the same typed error, never a silently-dropped candidate | Does not test two corrupted candidates in the same call (the first-encountered row's error suffices; row iteration order is not itself under test here) |
| `crates/jammi-db/tests/it/gang_membership.rs::keeper_reregisters_the_whole_membership_tuple_after_a_forced_delete` | P-M4 at the catalog layer, `peer_addr_of` too | Does not exercise the session-construction path (next file does) |
| `crates/jammi-db/tests/it/gang_membership.rs::prune_window_does_not_prune_a_member_merely_stale_within_the_window` | Both boundary directions in one test | Does not test EXACTLY at the window boundary |
| `crates/jammi-db/tests/it/member_root_constructor.rs::member_root_new_has_no_production_caller` | P-X4 (hardened, this revision): no `crates/<name>/src/**/*.rs` file in the whole workspace contains the literal call form `MemberRoot::new(` — a sanity floor on the walk itself (≥10 crates with a `src/` dir, ≥300 `.rs` files scanned) guards against a broken walk passing vacuously | Does not cover `benches/`/`examples/` (neither exists in this workspace today) or a call spelled through a type alias/re-export |
| `crates/jammi-db/src/config/tests.rs::from_config_member_root_is_resolved_result_root_verbatim_over_every_arm` | P-X1: the pure constructor's `member_root` == `resolved_result_root()` verbatim over the whole arm list, plus the "/" alone arm | Does not touch the filesystem — a real-session counterpart is `storage_root.rs` |
| `crates/jammi-db/src/config/tests.rs::from_config_never_aliases_gcs_and_gs_result_root_spellings` | P-X2 at the pure-function layer: the two spellings stay two different STRINGS on the row | Does not test admission (gang_membership.rs's renamed alias oracle proves the two strings are still gang members of each other, since `list_gang_members` never reads either) |
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
| `crates/jammi-db/src/catalog/instance.rs::tests::peer_addr_parses_host_port` / `::peer_addr_parse_keeps_ipv6_host_intact` / `::peer_addr_parses_a_full_bracketed_ipv6_host` / `::peer_addr_parses_a_dns_hostname` / `::peer_addr_refuses_no_colon` / `::_refuses_empty_host` / `::_refuses_zero_port` / `::_refuses_non_numeric_port` / `::_refuses_out_of_range_port` | Every `PeerAddr::parse` accept edge, including bracketed-IPv6-colon-preservation and a DNS hostname | Does not test a hostname requiring actual DNS resolution (this type never resolves) |
| `crates/jammi-db/src/catalog/instance.rs::tests::peer_addr_refuses_an_unbracketed_ipv6_literal` / `::peer_addr_refuses_an_unbracketed_loopback_ipv6_literal` (new, round-6, P-Y4) | An UNBRACKETED IPv6 host is refused, naming the refusal reason, for a full address and the loopback shorthand alike | Does not test a host with an embedded zone id (`fe80::1%eth0`) — not a shape this deployment's addresses use |
| `crates/jammi-ai/tests/it/instance_identity.rs::a_failed_first_upsert_worker_leaves_the_cell_none_so_the_keeper_writes_no_workers_row` (new, round-6, P-Y4) | An injected failure on the FIRST `upsert_worker` call leaves the registration's worker cell `None` (proven indirectly: a subsequent forced-delete + real keeper pass, which would write a `workers` row from a `Some` cell, writes none) | Does not directly read the cell (it is `pub(crate)`, unreachable from this external test crate) — the observable effect is the oracle |

---

## 9. Mutations executed / implied by the oracle shapes above

- **The self-exclusion / state-exclusion / kind-token-match conjuncts
  (NARROWED this revision — the root-byte-compare conjunct is DELETED, not
  merely untested; see the mutation below).** Each has its own isolated
  oracle (§8 table) holding every other conjunct at its satisfied value and
  flipping exactly one — deleting any single `continue` arm in
  `list_gang_members`'s Rust filter loop
  (`crates/jammi-db/src/catalog/jobs_repo.rs:2354-2367`) flips exactly the
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
  `crates/jammi-db/src/catalog/jobs_repo.rs:2154-2166` would flip
  `crates/jammi-ai/tests/it/instance_identity.rs::a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete`'s
  expected "no workers row after recovery" to a false resurrection.
- **The verbatim-identity property (of the VALUE — a scheme fold when
  WRITING it).** Re-introducing ANY scheme fold in
  `InstanceRegistration::from_config`
  (`crates/jammi-db/src/catalog/instance.rs:265-282`) — e.g. lower-casing or
  alias-folding the `member_root` string before wrapping it — is exactly
  what
  `crates/jammi-db/src/config/tests.rs::from_config_never_aliases_gcs_and_gs_result_root_spellings`
  is built to catch (the two spellings must stay distinct strings on the
  row).
- **P-Y1's own mutation, this revision: re-introducing the root conjunct at
  the JOIN layer.** Adding a root field back to `GangListing` and a
  `continue` arm comparing it in `list_gang_members`'s filter loop is
  exactly what
  `crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`,
  `::file_and_s3_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`,
  `::list_includes_members_despite_a_root_divergent_by_case_or_trailing_slash_root_is_not_consulted`,
  and
  `::list_includes_a_member_with_peer_addr_set_and_result_root_null_root_is_not_consulted`
  are built to catch: all four assert a member is returned DESPITE a root
  divergence that a reintroduced conjunct would exclude on.
- **P-Y4's mutation: reverting the cell-after-row order on the first
  `upsert_worker`.** Restoring `session.instance_registration().set_worker(...)`
  to BEFORE the `upsert_worker` call at
  `crates/jammi-ai/src/fine_tune/worker.rs:827-838` is exactly what
  `crates/jammi-ai/tests/it/instance_identity.rs::a_failed_first_upsert_worker_leaves_the_cell_none_so_the_keeper_writes_no_workers_row`
  is built to catch: a reverted ordering would leave the cell `Some` despite
  the injected upsert failure, and the keeper's later reregister would then
  write a `workers` row the test asserts must never appear.

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
- **Root identity across spellings, AND any membership predicate built on
  it at all** (WIDENED this revision, round-5 — §12) — filed as unit
  U5b-1a-A2 (§6), NOT a wave-3 precondition, and named as a precondition of
  U5b-1b-ii. `list_gang_members` in THIS unit consults no root whatsoever;
  `instances.result_root` is written verbatim and carried for A2's use.

---

## 11. Report

**Round-4 closing audit and the lead's stop-rule ruling (2026-09-15, from
the source design contract `CONTRACT-U5b-1a.md` §11).** Round 4 executed
every root-identity attack this unit's mechanism admits (config drift,
store-vs-row string, reregister/keeper, tenant scope, Postgres vs SQLite,
`MemberRoot::new` reachability under resolver 2, migration 035 bytes, the
P-X3 grep, the worker cell's lattice, `PeerAddr` sealing) and refuted every
one — the mechanism (§1–§5 above) stands unchanged since round 3. Round 4's
three BLOCKs and citation round 4's three findings were ALL documentation
of record, never the mechanism: the rigor record not yet committed (it is
the lead's export, committed LAST — see below); the wrapped path token at
(pre-fold) contract lines ~568–569 and its ~672 sibling, and an off-by-one
in the `MembershipConfig::validate` citation of §2 (it named the wrong
line for the `PeerAddr::parse` statement — corrected in §2 above, now
`:290` in this revision's tree, after the round-5 code commit shifted the
file by three lines); "canonical-root" wording surviving in
`deploy-server.md`, `reference-topologies.md`, `UNITS.md` (the §U5b-1a spec
and its `depends_on` line), `DESIGN.md`, `DIST-DATA-PLANE.md`, `DIST-r3.md`,
and `feat_500-C-U5a-1.md`; the maintainer guide's `MemberRoot::new(...)`
sentence (should read `MemberRoot::resolved(config)`). Advisories: the
`kinds` "otherwise producer-encoded" hatch on `WorkerFacts`'s field doc in
`instance.rs` (pre-fold, §1); the `advertising_config` fixture doc
describing a shape its body does not build; no oracle existed yet for the
corrupted-`peer_addr` typed-error arm
on either read verb; the doc sweep was keyed on literal vocabulary rather
than a machine check.

**The lead's ruling:** the round-4 pre-committed stop rule ("a BLOCK on ANY
finding touching the root column excises the column from the predicate")
is read as applying ONLY to a MECHANISM finding on the column — none of
round 4's findings was one; they are the round-3 excision's OWN
documentation fold left incomplete (P-X3/P-X4 as stated require every doc
site to carry the shipped words). The rule does NOT fire.

**Pre-committed for round 5 (binding):** a BLOCK of ANY kind on this unit
in round 5 — mechanism OR documentation of the root column — fires the
stop rule: the `result_root` column leaves the membership predicate
entirely (rows still carry it for display; the predicate becomes
`kinds`+`state`+`peer_addr` only), and the whole root-identity question
moves to U5b-1a-A2. There is no round 6.

**This revision (round 5) closes every item round 4 found:** the kinds
encoding is pinned at the writer with no "otherwise producer-encoded"
hatch (`instance.rs`, `jobs_repo.rs`); the `advertising_config` doc
describes its body; the corrupted-`peer_addr` arm is oracled on both read
verbs (§8 table, this revision); every doc/plan/contract site named above
is reworded to the shipped words (verbatim spelling, byte-equality,
necessary-never-sufficient) — see the fold commit's own sweep output; the
wrapped tokens and the `:139` off-by-one are corrected in THIS file,
re-derived against commit 1's (re-anchored) tree.

`impossibility_claims`:
- "no caller other than `InstanceRegistration::from_config` builds a value
  `Catalog::upsert_instance`/`Catalog::reregister_instance` accept" — not
  independently enumerated by a code-scanning oracle in this unit; the claim
  rests on the two verbs' own signatures (`&InstanceRegistration`) rather
  than an executed grep-shaped enumeration — `uncovered`.
- "a stored `peer_addr` that fails to parse is never silently treated as
  absent" — CLOSED (round-5): both `peer_addr_of` and `list_gang_members`
  return a typed `JammiError::Catalog` naming the instance on a parse
  failure, now with a manufactured corrupted row (an out-of-band `UPDATE
  instances SET peer_addr = 'not an addr'`, never through `PeerAddr::
  parse`) —
  `crates/jammi-db/tests/it/gang_membership.rs::
  peer_addr_of_returns_the_typed_error_for_a_corrupted_peer_addr` and
  `crates/jammi-db/tests/it/gang_membership.rs::
  list_gang_members_returns_the_typed_error_for_a_corrupted_peer_addr`
  (`test_case`-parameterized sqlite/postgres, `feature = "test-hooks"`) —
  no longer `uncovered`.
- "the `,` kind-encoding convention is honored by every real writer" —
  executed check: `crates/jammi-ai/src/fine_tune/worker.rs:829,836,866` are
  the ONLY three call sites of `self.kinds.join(",")` in this crate
  (confirmed by direct read, not a code-scanning oracle; re-derived this
  revision after the P-Y4 line shift) — `uncovered`.
- (round-3) "canonicalization ever made two spellings of one location
  provably identical to a third party" — this claim was never true (rounds
  1–2 canonicalized only the LOCAL `file://` case; a cloud root was already
  passed through with, at best, a scheme-alias fold that round 3 found
  itself unsound); the round-3 design does not attempt it — the necessity
  boundary (§ Scope) always covered this gap, now honestly with no
  machinery pretending otherwise.

`citations_reanchored`: every citation in §1–§5, §7–§9 of this revision was
re-derived directly against the tree of `13663cc178ab0f74a0f2811ff9aa725ba4a198b4`
(commit 1, re-anchored this round — see the header's round-6 doc-fold
re-anchor note) in this worktree by the writing agent; §6 (history) cites
no construct at all, by design (see §6's own header).

**Self-check (round-2 addendum requirement; re-run for the round-6
citation fold, at `13663cc178ab0f74a0f2811ff9aa725ba4a198b4`).** Before this
revision was committed, every `path:line`/`path:a-b` token AND every
`path::construct` token in §1–§5, §7–§9 (§6 excluded, cites none) was
machine-extracted (backtick-quoted `crates/…/*.rs::Construct` and
`crates/…/*.rs:N[-M]` patterns, scoped to each `## N.` heading's own span)
and checked against the tree of commit 1: a `path:line`/`path:a-b` token's
file was opened and its line count compared against the cited line(s); a
`path::construct` token's LAST path segment (after the final `::`, generics
stripped) was checked for literal occurrence in the named file.

```
python3 /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/12f161bf-7977-4318-a34d-d2eec24a0620/scratchpad/logs/db-selfcheck-r6.py
```

(a scratch script, never committed, written fresh this round — the round-5
`db-selfcheck-r5.py` script no longer exists in this scratchpad). **Result
(at commit 1 = `13663cc1`): 66 `path::construct` tokens checked, 66
resolved (their last segment occurs in the named file); 58 unique
`path:line`/`path:a-b` tokens checked, 58 resolved (the named file has at
least that many lines) — both counts 100%, zero unresolved.** (This is a
purely mechanical re-run of the round-5 script's own shape over the
CURRENT §1–§5/§7–§9 text; it does not independently resolve the bare
`:N`/`:N-M` shorthand citations that inherit their file from the
immediately preceding full `path:line` token, such as `:313`, `:308-311`,
and `:274` in §2 — those were checked by direct read instead, individually,
as part of the line-shift re-derivation above.)

---

## 12. Round-5 closing verdicts — the round-5 rule FIRES; round 6 is the LAST (this revision)

Round 5 (exhaustive) re-executed every mechanism attack from rounds 1–4 and refuted each;
citation round 5 PASSed. One finding stood: `docs/plans/68-compute-tier-substrate/units/DIST-DATA-PLANE.md:27`
(D9: `` `validate()`: `peer_advertise ⇒ peer_bind ⇒ result_root` ``) and
its sibling `:160` (`` "unit 2 enforces at startup … requires `storage.result_root`" ``) still
stated, in present tense, an `⇒ result_root` requirement this unit had already excised at round 3
— the round-4 wording sweep was keyed on `canonical*` and missed this REQUIREMENT class (`⇒`
chains, "enforces at startup"). §11's pre-committed round-5 rule ("a BLOCK of ANY kind — mechanism
OR documentation of the root column — fires the stop rule") FIRES. This revision (round 6, the
LAST — no round 7) is the fold that executes the firing:

- **P-Y1 (the predicate, mechanism).** `list_gang_members` admits on `kinds` (whole-token),
  `workers.state == Claiming`, `peer_addr` presence, freshness, and self-exclusion ONLY;
  `GangListing` carries NO root field at all (§1, §3). Oracle: two members rooted at `gcs://b/p`
  and `gs://b/p` (and, over an unrelated scheme pair, `file:///a` vs `s3://b`) ARE gang members of
  each other — the renamed tests say so in their own names
  (`crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`,
  `::file_and_s3_spelled_members_are_gang_members_of_each_other_root_is_not_consulted`) — and this
  contract states plainly (§ Scope, §3, §7 P-M3) that root identity is NOT part of admission in
  this unit.
- **P-Y2 (the row keeps the verbatim spelling, unchanged mechanism).** `instances.result_root` is
  still written as the byte-for-byte output of `resolved_result_root()`
  (`InstanceRegistration::from_config` → `MemberRoot::resolved`, §1–§2; P-X1's oracles are
  UNCHANGED — the pure-arm oracle
  `crates/jammi-db/src/config/tests.rs::from_config_member_root_is_resolved_result_root_verbatim_over_every_arm`,
  the four real-session oracles in `crates/jammi-ai/tests/it/storage_root.rs`, `MemberRoot::
  resolved` remains the ONE production constructor per P-X4). It is carried for U5b-1a-A2, which
  now owns the WHOLE root question — identity across spellings AND any membership predicate built
  on it — and is named as a precondition of U5b-1b-ii (gang formation):
  `docs/plans/67-distributed-training/README.md`'s unit table (§10),
  `docs/plans/67-distributed-training/UNITS.md`'s U5b-1b-ii `depends_on` line.
- **P-Y3 (documentation of record — the fold this revision performs).** Every sentence that
  said root equality is "necessary, never sufficient" for MEMBERSHIP now says instead: the row
  records the configured spelling verbatim; the membership predicate does not consult it in this
  unit; root identity and its predicate are U5b-1a-A2's question. Corrected, this commit:
  `docs/guide/src/configuration.md` (the `[server] peer_advertise` comment block),
  `docs/guide/src/security.md` (I-GANG's membership-predicate bullet),
  `docs/guide/src/deploy-server.md` (the `peer_advertise` paragraph),
  `docs/guide/src/reference-topologies.md`
  (Shape D's membership bullet), `docs/maintainer/MAINTAINER-GUIDE.md` §2.8b (the worker-cell
  ordering AND the `list_gang_members` predicate description),
  `docs/plans/67-distributed-training/UNITS.md` § U5b-1a (files_in_scope AND acceptance (a)) and its
  U5b-1b-ii `depends_on` line, `docs/plans/67-distributed-training/DESIGN.md` (the coordinator's
  `list_gang_members` call site — now filters on U5b-1a-A2's predicate itself, since
  `list_gang_members` no longer does), `docs/plans/67-distributed-training/README.md`'s
  U5b-1a-A2 row (widened: now root identity AND its predicate, a precondition of U5b-1b-ii),
  `docs/plans/68-compute-tier-substrate/units/DIST-DATA-PLANE.md:27` (D9, a dated correction —
  its bullet already carried the round-3 convention elsewhere, e.g. `:292`, which stands unedited)
  and `:160` (the precondition restated as unit 2's OWN requirement to enforce),
  `docs/plans/68-compute-tier-substrate/pressure/verdicts/DIST-r3.md:6` (a further dated correction appended
  after the existing round-3/4 one), `docs/rigor/contracts/feat_500-C-U5a-1.md:456` (the "coarse
  root pre-filter at listing time" phrase corrected — no such pre-filter exists), and this
  contract's own §1–§5/§7–§9 above. The class sweep this revision used, matching the lead's own
  anticipation:
  ```
  grep -rn '⇒' docs | grep -E 'peer_advertise|result_root|peer_bind'
  grep -rniE 'enforces at startup|refused at load|refuses at startup' docs
  grep -rniE 'result_root' docs crates --include='*.md' --include='*.rs' | grep -iE 'refus|requires|must exist|must be absolute|necessary'
  grep -rniE 'canonical[- ]?root|non-directory' docs crates --include='*.md' --include='*.rs'
  ```
  Every survivor after this commit is history (§6, DIST-r3.md's own dated-correction chain), a
  dated correction (D9, DIST-r3.md), or U5b-1a-A2's filing (README.md) — named individually above,
  none a live false claim — a claim round 6 REFUTED for five `crates/**` doc comments and one UNITS.md
  sentence, closed in §13 with the phrasing-independent sweep. The `⇒ result_root` and "enforces at startup" instances were the TWO
  the round-5 audit found; no third survivor of either pattern exists after this fold. This
  contract's own §1 fixture-list error (`crates/jammi-db/src/config/tests.rs` was listed as a
  `MemberRoot::new` call site; it never builds one — it reads `member_root` off a REAL
  `InstanceRegistration::from_config` result) is corrected in §1 above.
- **P-Y4 (advisories folded where cheap, mechanism).** `PeerAddr::parse` refuses an unbracketed
  IPv6 literal — `host.contains(':') && !(host.starts_with('[') && host.ends_with(']'))` is checked
  BEFORE the port parse (`crates/jammi-db/src/catalog/instance.rs:66-70`, inside `parse` at `:57`)
  — accepting a bracketed IPv6 host (`[::1]:9000`, `[2001:db8::1]:9000`), an IPv4 literal
  (`10.0.0.1`, port 9000), or a DNS hostname (`coordinator.internal`, port 9000), refusing an unbracketed one
  (`2001:db8::1:9000`, `::1:9000`) — oracles for each (§8 table, this revision).
  `crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_until`'s FIRST `upsert_worker` call
  (`:827-830`) now sets the registration's worker cell (`:832-838`) only on `Ok(())`; an `Err`
  (`:840-842`) leaves the cell untouched (`None`), gated by a NEW test-only failure-injection
  module mirroring `claim_test_hooks.rs`'s shape,
  `crates/jammi-db/src/catalog/worker_test_hooks.rs::arm_upsert_worker_failure` (`feature =
  "test-hooks"`), consumed once at the top of `Catalog::upsert_worker`
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::upsert_worker`, the `take_armed` check
  before the SQL). Oracle:
  `crates/jammi-ai/tests/it/instance_identity.rs::
  a_failed_first_upsert_worker_leaves_the_cell_none_so_the_keeper_writes_no_workers_row` — an
  armed failure, a real `EmbeddedWorker::spawn` with the worker gate closed (so `run_until` parks
  right after the failed attempt), and a subsequent forced-delete + one real `LeaseKeeper` pass:
  `list_workers()` never carries this instance, before OR after the keeper pass, proving the cell
  stayed `None` throughout (the direct cell read is unreachable from this external test crate —
  `pub(crate)` — so the observable effect on `workers` IS the oracle).

**Round 6 (this revision, FINAL — pre-committed by §11, honored here).** Closers run once on this
excision tip. A PASS ships. A BLOCK of any kind withholds the unit from wave 3: the branch is not
merged, the whole unit is refiled as U5b-1a-A3 with all six rounds as its spec, and U5b-1b-* wait
on it. No seventh round, no further excision — per the pre-committed rule, this stop applies
regardless of what a seventh round's finding would be.

---

## 13. Round-6 closing verdicts (2026-09-15) — the lead's takeover; the fold's last members; the worker cell as one fact

Round 6's closers: discipline PASS; citation BLOCK on one two-line drift (`prune_instances` cited at the line it
held at commit 1, two lines above its position after `b9c4c03f` grew a doc comment earlier in the file) and a
narrative count off by one (430 `.rs` files, not 429); audit BLOCK on (F1) five live statements — migration 035's
own rustdoc (two sentences), `crates/jammi-ai/src/session.rs::open_with_placement`'s doc and
`build_result_store`'s comment, and UNITS.md § U5b-1a's `MemberRoot` sentence — that still said the membership
predicate compares the root; (F2) §12's completeness claim, refuted by F1 (the round-5 sweep's patterns could not
match `compares`/`byte-for-byte`/`required`, and it never covered `crates/**/*.rs` doc comments); (F3) the worker
cell: after a failed first `upsert_worker` the loop fell through, the post-gate transition set the cell to
`claiming` UNCONDITIONALLY, and `set_worker_state`'s `Ok(false)` (no row) was discarded — a cell claiming a row
that did not exist, which a keeper reregister would then INSERT; the P-Y4 oracle opened the gate and stopped
without asserting, so it never entered that state. Advisory: `[]`, `[hello]` and bracketed zone-id hosts are
admitted unstated by `PeerAddr::parse`.

At this point the user took the unit over from the swarm ("implement the fixes, test and create the PRs and close
wave 3"), so the pre-committed round-6 withhold was NOT applied; the lead applied the fixes directly and the
closers were not re-run. What this revision changes:

- **The worker cell is one fact with its row.** Every lifecycle row write of the claim loop (`warming`, `claiming`,
  `draining`) goes through `crates/jammi-ai/src/fine_tune/worker.rs::write_worker_facts`: the cell is set FIRST to
  the facts about to be written (so a keeper reregister racing the write re-upserts exactly those facts, §8 B1),
  the row is written by `Catalog::upsert_worker` — an UPSERT, never a bare `UPDATE` — and on failure the cell is
  REVERTED to its previous snapshot (`None` after a failed first write). `Catalog::set_worker_state` keeps its own
  oracles (`jobs_queue.rs`) but has no caller on the loop. Oracle:
  `crates/jammi-ai/tests/it/instance_identity.rs::a_failed_first_upsert_worker_leaves_the_cell_none_so_the_keeper_writes_no_workers_row`
  now OPENS the gate after the armed failure, waits for the `claiming` row (created by the transition's upsert),
  force-prunes the instance again and asserts the keeper's reregister re-upserts exactly the cell's `state` and
  `kinds`.
- **The last five statements of the excised design are gone**, and the sweep that finds the class is the
  phrasing-independent one the audit used — `grep -rniE '(membership|gang|list_gang_members).{0,120}root|root.{0,120}(membership|gang member|list_gang_members)' crates docs`
  minus the negation vocabulary (`not consult|does not|no root|never|carries no|not part|A2|verbatim|history`)
  — plus `result_root × (compar|required|byte-for-byte)` over `crates/**/*.rs`. Executed at this revision: every
  survivor is a test name, a negation ("root is not consulted"), or `collective/`'s unrelated NCCL "root rank".
- The `prune_instances` citation is re-derived (`:2559`); the file count is 430.
- The IPv6 advisory is recorded here as a stated admitted domain: a bracketed host is validated for shape only
  (non-empty, no port inside the brackets); `[]`, `[hello]` and zone ids are admitted and fail at dial time, the
  same disclaimer `PeerAddr::parse`'s doc already makes for resolution.

Verification run by the lead at this revision: `cargo fmt`, `cargo clippy -p jammi-ai -p jammi-db --all-targets
-- -D warnings` (0), the extended oracle (ok), the four `…_root_is_not_consulted` oracles and the four IPv6
oracles (ok), the class sweep above (clean). The unit's full suites run again on the consolidated wave-3 branch
before its single PR.

## 14. Gates

`cargo fmt --all --check`; `cargo clippy -p jammi-db -p jammi-ai -p
jammi-server --all-targets -- -D warnings` (with AND without
`--features test-hooks` on `jammi-db`, since `MemberRoot::new` and the
`gang_membership.rs` module it is built through are `cfg`-gated on it —
`jammi-ai`'s own test targets already turn `jammi-db/test-hooks` on
unconditionally via its self dev-dependency, so only the `jammi-db`-alone
invocation needs the explicit flag both ways); `cargo test -p jammi-db -p
jammi-ai` (the plain, `test-hooks`-off run: `gang_membership.rs` is
`#[cfg(feature = "test-hooks")]`-gated at `tests/it/main.rs` and does not
compile in this pass, same as
`materialization_crash_recovery.rs`/`mutable_crash_recovery.rs`;
`member_root_constructor.rs`'s enumerating oracle is unconditional and DOES
run here) AND `cargo test -p jammi-db
--features test-hooks --test it` (the lane `gang_membership.rs` actually
runs in — `.github/workflows/ci.yml`'s "test-hooks lane" step, on every
PR) and `cargo test --workspace` (a server-startup-adjacent config change —
other crates' test harnesses spawn the binary with their own env); the
live-postgres lane (`--features live-postgres-tests`, `JAMMI_TEST_PG_URL`
set) for every `test_case`-parameterized test in
`crates/jammi-db/tests/it/gang_membership.rs` and
`crates/jammi-db/tests/it/migrations.rs`'s 035 oracle;
`python3 ci/scripts/check_doc_parity.py`;
`python3 ci/scripts/perf/check_citations.py`; `python3 ci/scripts/check_no_consumer_names.py`;
`python3 ci/scripts/check_swarm_bijection.py`; `RUSTDOCFLAGS="-D warnings"
cargo doc --no-deps -p jammi-db -p jammi-ai`.

## 15. The consolidated PR's Postgres lane (2026-09-15) — one shared database, row-scoped oracles

**What §14 listed and what actually ran.** §14 names the live-postgres lane
(`--features live-postgres-tests`, `JAMMI_TEST_PG_URL` set) as a gate for
every parameterized test in `crates/jammi-db/tests/it/gang_membership.rs`.
That lane was NOT executed on this machine before PR #579 opened — no local
Postgres existed — so CI's "Test (Postgres)" job was the arm's first
execution. It failed five `::postgres` cases at the consolidated tip
`0672f9ca`; the `::sqlite` arms of the same tests were green on every local
run because SQLite opens a fresh catalog per test.

**The defect (in the tests, not the verbs).** The lane runs the whole
`jammi-db` integration suite `--test-threads=1` against ONE persistent
database (`jammi_test_utils::unique_suffix`'s doc), twice (once more with
`test-hooks`). Two tests in this file were written as if each test owned the
table: (i) `list_gang_members_returns_the_typed_error_for_a_corrupted_peer_addr`
and `peer_addr_of_returns_the_typed_error_for_a_corrupted_peer_addr` planted
a row with `peer_addr = 'not an addr'` and returned without deleting it, so
every later `list_gang_members` in the lane surfaced that row's typed error
(four listing tests red, all naming the `corrupt-list-…` instance); (ii)
`prune_window_does_not_prune_a_member_merely_stale_within_the_window`
asserted `prune_instances(..) == 0` and then `== 1`, a count over the WHOLE
table that also counts every stale row a sibling test left behind (CI
observed 5; a second local run observed 24).

**The fix (this commit).** Both corrupt-row tests capture the verb's result,
`force_delete_instance` the poison row, and only then assert — on every arm,
so a red assertion never leaks the poison into later listings. The prune
oracle is row-scoped: after the in-window prune it asserts ITS row still
exists (`instance_row_exists`, a direct `SELECT … WHERE instance_id = $1`);
after the past-window prune it asserts the count is at least one and ITS row
is gone. The file header states the two disciplines the shared database
imposes (never assert a count over rows a test did not seed; delete a
predicate-intolerable row before asserting).

**Executed, both arms, both passes.** A PostgreSQL 16 cluster was stood up
under the session scratchpad (TCP only, port 54329) with the CI database
shape. At the pre-fix file: `cargo test -p jammi-db --features
live-postgres-tests,test-hooks --test it gang_membership -- --test-threads=1`
→ `39 passed; 5 failed` — the same five cases as CI, the same corrupted-row
message, the prune count `24 != 0`. At the fixed file: the `live-postgres-tests`
pass → `0` tests (the suite compiles in only with `test-hooks`, as CI's second
invocation does) and the `live-postgres-tests,test-hooks` pass → `44 passed;
0 failed` (22 sqlite + 22 postgres). The full lane (`--test it` over the
whole crate, both invocations, and the `jammi-server` introspection step) was
then run locally in the CI shape; its result is recorded in the PR.

**Residual.** Every other `test_case`-parameterized suite this wave touched
(`migrations.rs`'s 035 oracle, `member_root_constructor.rs`,
`memory_pool.rs`, `tenant_scope.rs`) seeds no row another test's predicate
could reject and asserts no whole-table count; the full-lane run above is the
executed check of that claim, not this sentence.
