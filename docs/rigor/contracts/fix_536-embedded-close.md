# CONTRACT — fix/536-embedded-close: a session the client opened is observable until it is closed

**Contract of record.** slug: `fix_536-embedded-close` · base: `main` @ `2055b33c` (merge-base of this
branch and `origin/main`) · this file is the committed mechanism contract `ci/scripts/check_rigor_record.py`
requires under `docs/rigor/contracts/**` before this unit's rigor record at
`docs/rigor/fix_536-embedded-close.jsonl` (the lead's export, landed separately) can satisfy that
checker's disclosure requirement — IF the checker ever arms for this unit. It does not: this branch's
`origin/main...HEAD` diff touches only `clients/python/**`, `cookbook/**`, `docs/rigor/contracts/**` (this
file) and `.jammi/escapes.jsonl` — none of `crates/**`, `ci/**`, `.github/workflows/**` — so
`check_rigor_record.py`'s own arming predicate (`ARMING_GLOBS`) never fires for this PR, and this contract
is written on the same footing as the record would be: for the human reviewer, not for a required check.
Citations below are tagged **(at the branch head)** — every `path:line` was re-opened and
re-verified at this worktree's current `HEAD` by this agent, and are named this way rather than by a
sha because a sha is rewritten by rebase (this file was first written at the commit whose subject is "test(cookbook): #536 leak guard subscribes to the client's session events — every construction route and alias shape, and a session collected without close, fail by name" (the events-subscription commit) and
re-anchored here after three further commits on this same branch moved line numbers in `conftest.py`,
`test_session_lifecycle_guard.py` and `test_session_registry.py`).

Owner: **docs-ci** (dispatched to write this contract only; the fix itself is `cookbook`/client-python
work already landed on this branch). Worktree `scratchpad/wt-536`, branch `fix/536-embedded-close`, at
the branch head (23 commits ahead of `main`, 6 behind `origin/main` per this branch's own stale local
`main` ref, as of the revision that counted them — the merge-base against the real `origin/main` is
`2055b33c`).

## The defect (esc-112, `.jammi/escapes.jsonl` id `esc-112-embedded-engine-outlives-its-tempdir`)
An embedded engine opened inside a `TemporaryDirectory` in a `cookbook/**` pytest lane could outlive the
directory: nothing in the test or in the suite's own fixtures forced `close()` to run before
`shutil.rmtree` (via pytest's own `tmp_path` teardown, or a script's manual cleanup) removed the backing
directory. Observed as `OSError: [Errno 66] Directory not empty` (macOS `ENOTEMPTY`) racing the still-open
engine's background worker, which keeps writing `catalog.db-wal`/`catalog.db-journal` into the directory
after `rmtree`'s `scandir` pass and before its `rmdir` — 42/80 runs at base `fe5ac560`, 16–21/80 across
three earlier verification rounds, 0/80 once every site closed its engine first (all runs executed on
macOS; Linux's `ENOTEMPTY` mapping, Errno 39, is the expected platform mapping, never a measured run —
`.jammi/escapes.jsonl:110`, `observable` field). Four narrower gate shapes were tried and each was
falsified by execution before this one shipped (see "Excision" below).

## Property (binding; quantified over EVERY session any construction route opens, in the pytest lanes)
Every session opened by a test under `cookbook/book/tests/**` is closed before that test's teardown
completes — detected **regardless of how the caller bound the name it called through** (`jammi.connect`
under any import/alias shape, or direct construction of the backend class), and regardless of whether the
session object itself is still reachable at the point of detection (a session dropped by refcount before
any teardown code runs must still be caught).

## Mechanism
### `clients/python/jammi/_sessions.py` — the live-session registry (183 lines at the branch head, unchanged since the events-subscription commit)
A process-wide, non-weak ledger keyed by a monotonic integer handle (`itertools.count`, never `id()` —
a collected-and-reused address would silently alias to an unrelated later object; guarded by
`test_handles_are_unique_across_sessions_even_after_collection`,
`clients/python/tests/test_session_registry.py:270-291` at the branch head):

- `register(session, label) -> int` (def at `clients/python/jammi/_sessions.py:74`, at the branch head): assigns the next handle,
  adds `session` to a `weakref.WeakSet` (`_live`, backing `open_sessions()`), records `{handle: label}` in
  a plain (non-weak) dict (`_open_ledger`, backing `open_session_labels()`), and fires every subscribed
  `on_register(handle, label)` listener synchronously, outside the lock.
- `unregister(session) -> None` (def at `:103`, at the branch head): pops the handle for `session`, removes it
  from the ledger and the `WeakSet`, and fires every `on_unregister(handle, label)` listener. A no-op
  (fires nothing) if `session` is not currently registered — closing twice, or closing after collection,
  stays safe.
- `open_sessions() -> Tuple[object, ...]` (def at `:120`) and `open_session_labels() -> Tuple[Tuple[int,
  str], ...]` (def at `:139`) are read-only snapshots; `observe(on_register, on_unregister) ->
  Callable[[], None]` (def at `:152`) subscribes and returns an idempotent unsubscriber. THREE of these
  five module-level functions — `open_sessions`, `open_session_labels`, `observe` — are re-exported at
  `clients/python/jammi/__init__.py:31` and named in `__all__` at `:51-53` (at the branch head); `register` and
  `unregister` stay private to this module (called only from the two `__init__`/`close()` seams below), so
  no external caller reaches them directly.

`register` is called as the **last statement** of `__init__` in each of the two resource-owning classes —
the one seam every construction route (`jammi.connect`, any direct backend construction, any future
`open`/`from_*` helper) passes through:

- `EmbeddedBackend.__init__` (`clients/python/jammi/_embedded.py:130-143`, at the branch head): line 143 is
  `self._session_handle = _register_session(self, label)`, where `label` is the resolved artifact
  directory (the `file://` target `_open_embedded` dispatches on; `""` for a direct construction that
  passes none).
- `RemoteDatabase.__init__` (`clients/python/jammi/_database.py:961-1006`, at the branch head): line 1006 is
  `self._session_handle = _register_session(self, endpoint)`, where `endpoint` is the printable remote
  target (not the original `target` string with its scheme — `_database.py`'s own `open_remote` strips
  that before this call).

`unregister` is called from `close()` in both classes: `EmbeddedBackend.close`
(`clients/python/jammi/_embedded.py:186-233`, at the branch head — `_unregister_session(self)` at `:233`, after the native handle's own
`close(release)` returns) and `RemoteDatabase.close` (`clients/python/jammi/_database.py:2848-2875`, at the branch head —
`_unregister_session(self)` at `:2875`, after the gRPC/Flight channels close and `self._closed = True`).
`tenant_scope()` on both backends yields the SAME already-registered instance, so it is not a separate
registration site (stated, not separately gated, in `clients/python/tests/test_session_registry.py:20-22`
at the branch head).

### `cookbook/book/tests/conftest.py::_no_leaked_sessions` — the runtime rail (228 lines at the branch head, 176 at the events-subscription commit)
An `autouse=True` fixture (def at `cookbook/book/tests/conftest.py:164`, at the branch head) that, for the duration of each test,
calls `jammi.observe(_on_register, _on_unregister)` (`:201`) to record every `(handle, label)` registered
and every handle unregistered during that test — synchronously, so a session dropped by refcount before
any teardown code runs is still recorded as registered. In its `finally` (`:204-228`), it unsubscribes
FIRST (`:205`, before computing the diff, so its own teardown never races a later listener call), then
computes `leaked = {handle: label for handle, label in registered.items() if handle not in
unregistered}` (`:206-210`) and, if non-empty, fails the test **by name** — `pytest.fail(message,
pytrace=False)` (`:228`) naming every leaked handle's label, unless the test already failed for another
reason (`:224-226`, in which case it only warns, so a leak inside an already-failing test does not bury
the true cause). This subscribes to the registry's *events*, never a liveness snapshot: `open_sessions()`
alone (a `WeakSet` view) is documented as unsound for exactly this shape (`_sessions.py`'s own module
docstring, lines 1-41 at the branch head, unchanged since the events-subscription commit) because a bare `jammi.connect(...)` statement, or a local dropped at
frame exit, is refcount-collected before any `finally`/teardown code runs, so a snapshot-diff guard would
see it as already closed.

**The capability arm** (new since the events-subscription commit, all at the branch head): the rail depends on the installed client
actually carrying the registry, not merely on `jammi` being importable. `_RAIL_ACTIVE =
jammi is not None and getattr(jammi, "observe", None) is not None` (`cookbook/book/tests/conftest.py:100`)
is computed once, at import, never per test. A session-scoped, `autouse=True` fixture,
`_warn_if_rail_inactive` (def at `:144`, body `:150-160`), fires exactly ONE `pytest.PytestWarning`
naming the installed `jammi.__version__` if `jammi` is present but `_RAIL_ACTIVE` is false — the shape a
client built before the registry shipped takes (the nightly release-recipe leg that installs a previously
published wheel rather than HEAD source, per `.github/workflows/cookbook-render.yml`, which pins nothing).
`_no_leaked_sessions` itself checks `_RAIL_ACTIVE` first (`:181`) and, if false, `yield`s and `return`s
(`:185-186`) without subscribing — so a client missing `jammi.observe` cannot raise `AttributeError` out of
every test's setup; the whole session-leak rail is simply inactive for the run, once warned about, rather
than a per-test crash.

## Oracles (by name, each with what it excludes)
- **`clients/python/tests/test_session_registry.py`** (328 lines at the branch head, 329 at the events-subscription commit — a
  docstring rewrite dropped one review-process line, no test moved otherwise): per-route
  register/unregister round-trips for `jammi.connect("file://…")`, direct `EmbeddedBackend` construction,
  `jammi.connect("grpc://…")` and direct `RemoteDatabase` construction
  (`test_connect_file_route_appears_and_disappears`,
  `test_direct_embedded_backend_construction_appears_and_disappears`,
  `test_connect_grpc_route_appears_and_disappears`,
  `test_direct_remote_database_construction_appears_and_disappears`, `:75-91`, `:113-133` at the branch head) —
  each excludes the OTHER route, so no single test's pass certifies the shared seam; together they cover
  every constructor route named in the mechanism section. `test_embedded_unclosed_session_disappears_once_collected`
  / `test_remote_unclosed_session_disappears_once_collected` (`:102-107`, `:146-153`) exclude the
  event/ledger property entirely — they assert only the `WeakSet` view's collection behavior, which this
  same file's own docstring (`:24-31`) states is NOT what a leak guard can rely on.
  `test_dropped_without_close_leaves_a_register_with_no_unregister` (`:230-253`) is the property the rail
  actually needs: a bare, unbound `jammi.connect(...)` statement still leaves a register event with no
  matching unregister, findable in `open_session_labels()` after the object itself is gone — it excludes
  every route that keeps a live reference (already covered by the round-trip tests above).
  `test_handles_are_unique_across_sessions_even_after_collection` (`:270-291`) excludes correctness of
  labels or events entirely; it is a handle-collision oracle only, over 200 construct/drop/collect cycles
  (a single cycle would only sometimes reuse a freed address — looping makes an `id()`-backed scheme fail
  with overwhelming reliability while a monotonic counter never repeats). `test_unsubscribe_stops_delivery`
  (`:294-302`) and `test_concurrent_open_close_produce_balanced_events` (`:305-328`) exclude the
  registration seam entirely; they check only `observe()`'s own subscription contract.
- **`cookbook/book/tests/test_session_lifecycle_guard.py`** (519 lines at the branch head, 376 at the events-subscription commit — two
  test functions and the capability suite below were added by later commits on this branch): the runtime
  rail's own non-vacuity control, run as a REAL pytest session via `pytester` against the actual committed
  `conftest.py` (read from disk, `:45`) — no test in this file exercises a helper function directly,
  because the guard's own subject is what the OUTER pytest run reports (exit code, ERROR summary, message
  text), which excludes nothing smaller could assert honestly.
  - The two **transport fixtures** — `test_leaked_session_fails_by_name_on_both_transports` (`:74-112`):
    an embedded leak (bare `jammi.connect(f"file://{tmp_path}")`, no binding held) and a remote leak
    (bare `jammi.connect("grpc://<loopback>")` on the conventional local port, a lazy channel needing no server) each produce exactly
    one teardown ERROR, named (`:99-104`), with the label present in the message (`:110-112`); excludes
    every alias/construction shape below (a single pair of transports only).
  - The **closing control** — `test_closing_control_passes` embedded in `_THROWAWAY_SUITE` (`:65-70`) and
    asserted via `result.assert_outcomes(passed=3, errors=2, failed=0)` (`:96`): proves the guard does not
    fail a test merely for having opened a session; excludes the leak-detection path entirely.
  - The **16-shape leak fixture** — `test_every_alias_and_construction_shape_leaks_by_name`
    (`:318-348`), driving `_SHAPES_SUITE` (`:150-292`): the 13 import-time binding shapes audit #5 found
    (class-body alias, `try:`/`if:`-nested import, parenthesised multi-line import, star import,
    module-alias-then-attribute, `getattr`, `importlib.import_module`, tuple-unpack, default-arg,
    aliased import, underscore bare assignment, plain column-zero import, and a local dropped at frame
    exit) plus both backends' direct construction — 16 leaking tests named in `_LEAKING_SHAPE_TESTS`
    (`:298-315`) — each individually asserted present under an `ERROR` line (`:342-344`), with
    `test_shape_closing_control_passes` (`:286-291`) asserted ABSENT from any ERROR line (`:347-348`);
    excludes the subdirectory-collection shape and the two-transport-only case above (disjoint fixtures).
  - **`test_leak_inside_an_already_failing_test_is_warned_not_failed_again`** (`:351-416`, new since
    the events-subscription commit): the guard's OTHER arm — a test that fails on its own assertion AND also leaves a session
    open is reported as exactly one `failed` (the assertion, at CALL) plus ONE `PytestWarning` naming the
    leaked label (`result.assert_outcomes(failed=1, errors=0, passed=0, warnings=1)` at `:384`, the label
    check at `:388`), never a second `error` piled on top of the true cause — the short-summary line count
    is asserted to stay at exactly one, and that one line must start with `FAILED`, never `ERROR`
    (`:407-416`). A leak inside an already-failing test is a warning, never a second failure — this test is
    the only oracle for that branch; it excludes the mirror case (the identical leak inside an otherwise
    PASSING test), already covered above by `test_leaked_session_reports_exactly_once_by_name`
    (`:115-134`), where the leak itself IS the sole reported problem.
  - The **subdirectory fixture** —
    `test_leaked_session_in_a_tests_subdirectory_module_is_failed_by_name` (`:419-442`): a leak inside
    `pytester.mkpydir("nested_suite")/test_nested_leak.py` (`:428-436`) is still failed by name
    (`:442`) — excludes every shape/transport above; it is a collection-scope oracle only (the deleted
    static alias gate's documented scope limit was one directory level; this rail is not walking files at
    all, so it has none).
  - **`test_rail_inactive_without_the_registry_warns_once_and_runs_clean`** (`:492-519`, new since
    the events-subscription commit, F1): exercises the capability arm (`cookbook/book/tests/conftest.py`'s `_RAIL_ACTIVE` at
    `:100`, the session-scoped `_warn_if_rail_inactive` fixture at `:144-160`, and the early return in
    `_no_leaked_sessions` at `:181-186`) against a minimal fake `jammi` — `_FAKE_PRE_REGISTRY_JAMMI`
    (`:454-468`) exposes `connect` but no `observe`, standing in for a client built before the registry
    shipped. The run reports zero setup errors, including for the one test that actually leaks
    (`result.assert_outcomes(passed=3, errors=0, failed=0, warnings=1)` at `:514`, `result.ret == 0` at
    `:515`), and exactly ONE warning naming the fake version (`:518-519`). A client without the registry
    runs the suite with the rail inactive and ONE visible warning — the published-wheel lane (the nightly
    release-recipe leg pinning nothing, per `.github/workflows/cookbook-render.yml`) lags HEAD by
    construction, so this is the expected shape, not a bug; excludes every leak-detection property above
    entirely — a leak under this fake client is invisible by construction (`test_leaks_but_the_rail_is_inactive`,
    `:483-489`), not caught and reported.

## The excision (no static shape gate ships)
Four narrower mechanisms were tried, on this same branch, and each was falsified by execution before this
one shipped: (1) a whole-file textual close-oracle (block on `_offending_sites` crediting a close from an
unrelated function); (2) an exit-path-aware oracle scoped to `with`-item/`finally` shapes (block on a
`finally:` close credited from inside a comment, an `if`, a nested `def`, or a sibling `finally` after a
connect-in-a-loop); (3) a with-item-only oracle (block on a with-item whose context manager is not the
session, an `ExitStack` built outside the block, two tempdir items on one physical line, a >20-line
parenthesised-header bail-out, and any dedented line inside the block — each passing with zero offenders
while an independent AST shadow oracle found 0 live offenders on the real tree, proving the sites were
fixed but the gate was not sound); (4) a regex alias gate over `cookbook/book/tests/**` requiring
module-attribute binding (block on 10 of 13 import-time binding shapes bypassing both the runtime guard
and the alias gate's own negative controls). The registry+events mechanism in this file replaces all four:
it does not read source text or a bound name at all, so no enumeration of binding shapes remains as a
precondition. The real exit-path control-flow analysis that would let a STATIC gate credit a close
soundly — the capability none of the four attempts had — is filed as its own unit,
[issue #539](https://github.com/f-inverse/jammi-ai/issues/539), which owns only the non-pytest lanes
(scripts, recipes, quickstart, the executed chapter cells) that this runtime rail cannot reach because it
is a pytest fixture, not a file walker (`cookbook/book/tests/conftest.py:76-78`, at the branch head, states this
limit by name — see also "Stated limits" below).

## Stated limits (this rail's own scope, not a defect)
The rail observes sessions constructed inside a test's OWN function-scoped fixture window: it subscribes
at that test's own setup and reads the diff in its own `finally`, before any coarser fixture tears down
(`cookbook/book/tests/conftest.py:64-69`, at the branch head). The suite rule this actually enforces is narrower
than "every session is eventually closed" — it is that **a test-opened session is closed by that SAME
test** (`:67-69`). Two shapes fall outside this window by construction, both filed as
[issue #552](https://github.com/f-inverse/jammi-ai/issues/552) (`:69-76`): a session registered before
this fixture subscribes at all (constructed at import time, or in a module- or session-scoped fixture's
own setup), and a session that SPANS tests (opened by one test, left open past that test's own teardown,
and only closed later by a different test or a coarser fixture) — the latter is reported, if it is
reported at all, against the test that OPENED it, never the one that eventually closes it. That same
citation also files the registry ledger's unbounded size (nothing ever prunes `_open_ledger` for a session
that is never closed and never collected) and the `label=""` default for a direct construction that passes
none, under the same issue. What this rail does NOT reach AT ALL is the non-pytest lanes — scripts,
recipes, quickstart, the executed chapter cells — filed as
[issue #539](https://github.com/f-inverse/jammi-ai/issues/539) (`:76-78`), because it is a pytest fixture,
not a file walker; a static gate over those lanes is a separate unit, not a narrower version of this one.

## The site set (esc-112's resolution names it by subject, never a bare count)
`.jammi/escapes.jsonl:110` (`esc-112-embedded-engine-outlives-its-tempdir`, `resolution` field) names,
by commit subject rather than by count or by sha (shas are rewritten by rebase and trailer amends): the
original site fixes across `cookbook/book/scripts/**`, `cookbook/book/tests/test_rails.py` and
`test_unified_client_cache.py`, `cookbook/quickstart/quickstart.py` and the recipes; the three
`with`-item conversions (`build_cdc_cache.py`, `build_feature_store_cache.py`,
`build_tenancy_cache.py`); the `build_segmented_ann_cache.py` and `recompute.qmd` chapter-cell fixes and
`build_point_in_time_cache.py`'s dropped leaked `mkdtemp`; the conftest rail and its non-vacuity control;
and the registry+events mechanism and its two test modules described above. It states explicitly which
earlier-round artifacts (the four excised gate shapes and their own oracle logic) no longer hold and are
superseded, not retained.

## Mutations executed against this mechanism (each with the oracle that dies)
- **Ledger insert dropped** (`_open_ledger[handle] = label` removed from `register`): every
  `open_session_labels()` assertion in `test_session_registry.py` (`:251`, `:263-264`) and every leak
  message's label lookup in `cookbook/book/tests/conftest.py:213-216` fails — the guard would still report a leak count but
  never a name.
- **`id()`-based handles instead of the monotonic counter**: `test_handles_are_unique_across_sessions_even_after_collection`
  (`clients/python/tests/test_session_registry.py:270-291`) fails with overwhelming reliability within 200 iterations (a freed
  address is the next one CPython's per-size-class freelist hands back).
- **`unregister` removed from `close()`** (either backend): every "…appears_and_disappears" test
  (`clients/python/tests/test_session_registry.py:75-91`, `:113-133`, `:127-133`) fails its post-`close()` assertion, and
  every closing-control test in `test_session_lifecycle_guard.py` (`:65-70`, `:286-291`) starts failing
  because the closed session is now reported as a leak.
- **Registration moved off the shared constructor** (e.g. into `_open_embedded`/`open_remote` instead of
  `EmbeddedBackend.__init__`/`RemoteDatabase.__init__`): `test_direct_embedded_backend_construction_appears_and_disappears`
  and `test_direct_remote_database_construction_appears_and_disappears`
  (`clients/python/tests/test_session_registry.py:87-91`, `:127-133`) fail, and
  `test_shape_direct_embedded_construction`/`test_shape_direct_remote_construction`
  (`cookbook/book/tests/test_session_lifecycle_guard.py:265-284`) stop being reported as leaks at all — the completeness
  claim in `test_every_alias_and_construction_shape_leaks_by_name` (`:318-348`) fails on those two names.
- **The guard's fail neutered** (`pytest.fail` replaced with a no-op or `request.node.warn` on every
  path in `cookbook/book/tests/conftest.py:212-228`): every ERROR assertion across
  `test_session_lifecycle_guard.py` (`:96`, `:131`, `:334-344`, `:440-442`) fails — the runs report
  `errors=0` where the tests expect leaks named.
- **Unsubscribe before yield** (moving `unsubscribe()`, `cookbook/book/tests/conftest.py:205`, to before `yield` at `:202-203`
  instead of in `finally` after it): the fixture stops observing any event fired during the test body
  itself, so every leak in `test_session_lifecycle_guard.py` goes unreported — `errors=0` everywhere a
  leak is expected.
- **Matching inverted** (`leaked` computed as registered ∩ unregistered instead of registered −
  unregistered, `cookbook/book/tests/conftest.py:206-210`): every closing-control test starts failing (a properly closed
  session is now reported as the "leak") and every genuine leak stops being reported — both directions of
  `test_leaked_session_fails_by_name_on_both_transports`'s and
  `test_every_alias_and_construction_shape_leaks_by_name`'s outcome assertions (`:96`, `:334-338`) fail.
- **Capability arm removed** (`_RAIL_ACTIVE`'s check deleted from `_no_leaked_sessions`,
  `cookbook/book/tests/conftest.py:181-186`, so the fixture always calls `jammi.observe(...)`
  unconditionally): `test_rail_inactive_without_the_registry_warns_once_and_runs_clean`
  (`cookbook/book/tests/test_session_lifecycle_guard.py:492-519`) fails — every test's setup against the
  fake pre-registry `jammi` now raises `AttributeError: module 'jammi' has no attribute 'observe'`, so
  `result.assert_outcomes(passed=3, errors=0, ...)` at `:514` sees setup errors instead, and `result.ret
  == 0` at `:515` fails too.
- **Warning suppressed** (`request.node.warn(pytest.PytestWarning(message))` at
  `cookbook/book/tests/conftest.py:226` replaced with a no-op, or the whole `if`/`else` at `:224-228`
  collapsed to nothing on the already-failed branch): `test_leak_inside_an_already_failing_test_is_warned_not_failed_again`'s
  `warnings=1` assertion (`cookbook/book/tests/test_session_lifecycle_guard.py:384`) fails, and `assert
  "left 1 jammi session(s) open" in full` (`:388`) fails too — the leak is dropped silently rather than
  reported at all, once the test that leaked it has already failed.
- **Warn-arm inverted** (the `if request.session.testsfailed > failed_before` / `else` branches at
  `cookbook/book/tests/conftest.py:224-228` swapped, so an already-failing test now takes the
  `pytest.fail` branch and a cleanly-failing-nothing test takes the `warn` branch — both branches fail):
  `test_leak_inside_an_already_failing_test_is_warned_not_failed_again`'s `errors=0` (`:384`) fails (the
  guard now piles a second `error` on top of the test's own `FAILED`), and
  `test_leaked_session_reports_exactly_once_by_name`'s `errors=1` (`:131`) flips to a mere warning, so its
  own outcome assertion fails too.
