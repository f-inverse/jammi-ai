# CONTRACT — fix/536-embedded-close: a session the client opened is observable until it is closed

**Contract of record.** slug: `fix_536-embedded-close` · base: `main` @ `2055b33c` (merge-base of this
branch and `origin/main`) · this file is the committed mechanism contract `ci/scripts/check_rigor_record.py`
requires under `docs/rigor/contracts/**` before this unit's rigor record at
`docs/rigor/fix_536-embedded-close.jsonl` (the lead's export, landed separately) can satisfy that
checker's disclosure requirement — IF the checker ever arms for this unit. It does not: this branch's
`origin/main...HEAD` diff touches only `clients/python/**`, `cookbook/**` and `.jammi/escapes.jsonl` —
none of `crates/**`, `ci/**`, `.github/workflows/**` — so `check_rigor_record.py`'s own arming predicate
(`ARMING_GLOBS`) never fires for this PR, and this contract is written on the same footing as the record
would be: for the human reviewer, not for a required check. Citations below are tagged **(at 47613450)**
— the commit this worktree's `HEAD` sits at when this file was written; every `path:line` was opened at
that sha by this agent.

Owner: **docs-ci** (dispatched to write this contract only; the fix itself is `cookbook`/client-python
work already landed on this branch). Worktree `scratchpad/wt-536`, branch `fix/536-embedded-close`, HEAD
`47613450` (19 commits ahead of `main`, 6 behind `origin/main` per this branch's own stale local `main`
ref — the merge-base against the real `origin/main` is `2055b33c`).

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
### `clients/python/jammi/_sessions.py` — the live-session registry (183 lines at 47613450)
A process-wide, non-weak ledger keyed by a monotonic integer handle (`itertools.count`, never `id()` —
a collected-and-reused address would silently alias to an unrelated later object; guarded by
`test_handles_are_unique_across_sessions_even_after_collection`,
`clients/python/tests/test_session_registry.py:271-292` at 47613450):

- `register(session, label) -> int` (def at `clients/python/jammi/_sessions.py:74`, at 47613450): assigns the next handle,
  adds `session` to a `weakref.WeakSet` (`_live`, backing `open_sessions()`), records `{handle: label}` in
  a plain (non-weak) dict (`_open_ledger`, backing `open_session_labels()`), and fires every subscribed
  `on_register(handle, label)` listener synchronously, outside the lock.
- `unregister(session) -> None` (def at `:103`, at 47613450): pops the handle for `session`, removes it
  from the ledger and the `WeakSet`, and fires every `on_unregister(handle, label)` listener. A no-op
  (fires nothing) if `session` is not currently registered — closing twice, or closing after collection,
  stays safe.
- `open_sessions() -> Tuple[object, ...]` (def at `:120`) and `open_session_labels() -> Tuple[Tuple[int,
  str], ...]` (def at `:139`) are read-only snapshots; `observe(on_register, on_unregister) ->
  Callable[[], None]` (def at `:152`) subscribes and returns an idempotent unsubscriber. All four are
  re-exported at `clients/python/jammi/__init__.py:31` and named in `__all__` at `:51-53` (at 47613450).

`register` is called as the **last statement** of `__init__` in each of the two resource-owning classes —
the one seam every construction route (`jammi.connect`, any direct backend construction, any future
`open`/`from_*` helper) passes through:

- `EmbeddedBackend.__init__` (`clients/python/jammi/_embedded.py:130-143`, at 47613450): line 143 is
  `self._session_handle = _register_session(self, label)`, where `label` is the resolved artifact
  directory (the `file://` target `_open_embedded` dispatches on; `""` for a direct construction that
  passes none).
- `RemoteDatabase.__init__` (`clients/python/jammi/_database.py:961-1006`, at 47613450): line 1006 is
  `self._session_handle = _register_session(self, endpoint)`, where `endpoint` is the printable remote
  target (not the original `target` string with its scheme — `_database.py`'s own `open_remote` strips
  that before this call).

`unregister` is called from `close()` in both classes: `EmbeddedBackend.close`
(`clients/python/jammi/_embedded.py:186-233`, at 47613450 — `_unregister_session(self)` at `:233`, after the native handle's own
`close(release)` returns) and `RemoteDatabase.close` (`clients/python/jammi/_database.py:2848-2875`, at 47613450 —
`_unregister_session(self)` at `:2875`, after the gRPC/Flight channels close and `self._closed = True`).
`tenant_scope()` on both backends yields the SAME already-registered instance, so it is not a separate
registration site (stated, not separately gated, in `clients/python/tests/test_session_registry.py:20-22`
at 47613450).

### `cookbook/book/tests/conftest.py::_no_leaked_sessions` — the runtime rail (176 lines at 47613450)
An `autouse=True` fixture (def at `cookbook/book/tests/conftest.py:114`, at 47613450) that, for the duration of each test,
calls `jammi.observe(_on_register, _on_unregister)` (`:149`) to record every `(handle, label)` registered
and every handle unregistered during that test — synchronously, so a session dropped by refcount before
any teardown code runs is still recorded as registered. In its `finally` (`:152-176`), it unsubscribes
FIRST (`:153`, before computing the diff, so its own teardown never races a later listener call), then
computes `leaked = {handle: label for handle, label in registered.items() if handle not in
unregistered}` (`:154-158`) and, if non-empty, fails the test **by name** — `pytest.fail(message,
pytrace=False)` (`:176`) naming every leaked handle's label, unless the test already failed for another
reason (`:172-174`, in which case it only warns, so a leak inside an already-failing test does not bury
the true cause). This subscribes to the registry's *events*, never a liveness snapshot: `open_sessions()`
alone (a `WeakSet` view) is documented as unsound for exactly this shape (`_sessions.py`'s own module
docstring, lines 1-30 at 47613450) because a bare `jammi.connect(...)` statement, or a local dropped at
frame exit, is refcount-collected before any `finally`/teardown code runs, so a snapshot-diff guard would
see it as already closed.

## Oracles (by name, each with what it excludes)
- **`clients/python/tests/test_session_registry.py`** (329 lines at 47613450): per-route register/unregister
  round-trips for `jammi.connect("file://…")`, direct `EmbeddedBackend` construction, `jammi.connect("grpc://…")`
  and direct `RemoteDatabase` construction (`test_connect_file_route_appears_and_disappears`,
  `test_direct_embedded_backend_construction_appears_and_disappears`,
  `test_connect_grpc_route_appears_and_disappears`,
  `test_direct_remote_database_construction_appears_and_disappears`, `:76-92`, `:114-134` at 47613450) —
  each excludes the OTHER route, so no single test's pass certifies the shared seam; together they cover
  every constructor route named in the mechanism section. `test_embedded_unclosed_session_disappears_once_collected`
  / `test_remote_unclosed_session_disappears_once_collected` (`:103-108`, `:147-154`) exclude the
  event/ledger property entirely — they assert only the `WeakSet` view's collection behavior, which this
  same file's own docstring (`:24-32`) states is NOT what a leak guard can rely on.
  `test_dropped_without_close_leaves_a_register_with_no_unregister` (`:231-254`) is the property the rail
  actually needs: a bare, unbound `jammi.connect(...)` statement still leaves a register event with no
  matching unregister, findable in `open_session_labels()` after the object itself is gone — it excludes
  every route that keeps a live reference (already covered by the round-trip tests above).
  `test_handles_are_unique_across_sessions_even_after_collection` (`:271-292`) excludes correctness of
  labels or events entirely; it is a handle-collision oracle only, over 200 construct/drop/collect cycles
  (a single cycle would only sometimes reuse a freed address — looping makes an `id()`-backed scheme fail
  with overwhelming reliability while a monotonic counter never repeats). `test_unsubscribe_stops_delivery`
  (`:295-303`) and `test_concurrent_open_close_produce_balanced_events` (`:306-329`) exclude the
  registration seam entirely; they check only `observe()`'s own subscription contract.
- **`cookbook/book/tests/test_session_lifecycle_guard.py`** (376 lines at 47613450): the runtime rail's
  own non-vacuity control, run as a REAL pytest session via `pytester` against the actual committed
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
    (`:320-350`), driving `_SHAPES_SUITE` (`:152-294`): the 13 import-time binding shapes audit #5 found
    (class-body alias, `try:`/`if:`-nested import, parenthesised multi-line import, star import,
    module-alias-then-attribute, `getattr`, `importlib.import_module`, tuple-unpack, default-arg,
    aliased import, underscore bare assignment, plain column-zero import, and a local dropped at frame
    exit) plus both backends' direct construction — 16 leaking tests named in `_LEAKING_SHAPE_TESTS`
    (`:300-317`) — each individually asserted present under an `ERROR` line (`:344-346`), with
    `test_shape_closing_control_passes` (`:288-293`) asserted ABSENT from any ERROR line (`:348-350`);
    excludes the subdirectory-collection shape and the two-transport-only case above (disjoint fixtures).
  - The **subdirectory fixture** —
    `test_leaked_session_in_a_tests_subdirectory_module_is_failed_by_name` (`:353-377`): a leak inside
    `pytester.mkpydir("nested_suite")/test_nested_leak.py` (`:362-370`) is still failed by name
    (`:376`) — excludes every shape/transport above; it is a collection-scope oracle only (the deleted
    static alias gate's documented scope limit was one directory level; this rail is not walking files at
    all, so it has none).

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
is a pytest fixture, not a file walker (`cookbook/book/tests/conftest.py:54-56`, at 47613450, states this
limit by name).

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
  `open_session_labels()` assertion in `test_session_registry.py` (`:252`, `:264-265`) and every leak
  message's label lookup in `cookbook/book/tests/conftest.py:161-163` fails — the guard would still report a leak count but
  never a name.
- **`id()`-based handles instead of the monotonic counter**: `test_handles_are_unique_across_sessions_even_after_collection`
  (`clients/python/tests/test_session_registry.py:271-292`) fails with overwhelming reliability within 200 iterations (a freed
  address is the next one CPython's per-size-class freelist hands back).
- **`unregister` removed from `close()`** (either backend): every "…appears_and_disappears" test
  (`clients/python/tests/test_session_registry.py:76-92`, `:114-134`, `:128-134`) fails its post-`close()` assertion, and
  every closing-control test in `test_session_lifecycle_guard.py` (`:65-70`, `:288-293`) starts failing
  because the closed session is now reported as a leak.
- **Registration moved off the shared constructor** (e.g. into `_open_embedded`/`open_remote` instead of
  `EmbeddedBackend.__init__`/`RemoteDatabase.__init__`): `test_direct_embedded_backend_construction_appears_and_disappears`
  and `test_direct_remote_database_construction_appears_and_disappears`
  (`clients/python/tests/test_session_registry.py:88-92`, `:128-134`) fail, and
  `test_shape_direct_embedded_construction`/`test_shape_direct_remote_construction`
  (`cookbook/book/tests/test_session_lifecycle_guard.py:267-286`) stop being reported as leaks at all — the completeness
  claim in `test_every_alias_and_construction_shape_leaks_by_name` (`:320-350`) fails on those two names.
- **The guard's fail neutered** (`pytest.fail` replaced with a no-op or `request.node.warn` on every
  path in `cookbook/book/tests/conftest.py:160-176`): every ERROR assertion across
  `test_session_lifecycle_guard.py` (`:96`, `:131`, `:336-346`, `:374`) fails — the runs report
  `errors=0` where the tests expect leaks named.
- **Unsubscribe before yield** (moving `unsubscribe()`, `cookbook/book/tests/conftest.py:153`, to before `yield` at `:150-151`
  instead of in `finally` after it): the fixture stops observing any event fired during the test body
  itself, so every leak in `test_session_lifecycle_guard.py` goes unreported — `errors=0` everywhere a
  leak is expected.
- **Matching inverted** (`leaked` computed as registered ∩ unregistered instead of registered −
  unregistered, `cookbook/book/tests/conftest.py:154-158`): every closing-control test starts failing (a properly closed
  session is now reported as the "leak") and every genuine leak stops being reported — both directions of
  `test_leaked_session_fails_by_name_on_both_transports`'s and
  `test_every_alias_and_construction_shape_leaks_by_name`'s outcome assertions (`:96`, `:336-340`) fail.
