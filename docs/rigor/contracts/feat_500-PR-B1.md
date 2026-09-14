# CONTRACT — feat/500-PR-B1: a paid gang leg proves what it claims, and a materialised training set is anchor-faithful

**Contract of record.** slug: `feat_500-PR-B1` · this file is the committed mechanism contract
`ci/scripts/check_rigor_record.py` requires under `docs/rigor/contracts/**` before this unit's rigor
record at `docs/rigor/feat_500-PR-B1.jsonl` (the lead's export, landed separately) satisfies that
checker's disclosure requirement. PR-B1 consolidates two units that have not yet been merged into one
branch: **U7b-A1-pull** (branch `feat/500-B-gang-local`, head `288a34c7`) and **U2a** (branch
`feat/500-B-U2a`, head `06720d1e`). Every citation below names a construct by path and item
(`crate/path::item` for Rust, `ci/scripts/<file>.py::<function>` for Python, an enclosing named function
for shell where one exists) — never a `path:line` offset, which moves under an unrelated edit and which
this contract's own checker cannot re-verify against content, only against a file's total length. Where
a fact has no addressable symbol (a shell script's top-level statements, a YAML/Markdown sentence), the
exact command this agent ran and its output are shown in a fenced block instead of an asserted line
number. Both branches share the U7a/U4a/release-claim-spin merge line: A1-pull's own base is `9af28de0`
(the merge of U7a + U4a + the release-claim-spin fix into `feat/500-B-gang-local`); U2a's own base is
`9db8d395` (the release-claim-spin fix's own head, one commit earlier on the same line). Neither unit's
diff touches a file the other unit touches — A1-pull's `9af28de0..288a34c7` diff is confirmed scoped to
`.docker/ci.Dockerfile`, `.github/workflows/{ci,gpu-gang}.yml`, `ci/scripts/**`,
`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs` and `docs/{maintainer/dev-gpu.md,
plans/67-distributed-training/UNITS.md}`; U2a's diff is confirmed scoped to `crates/jammi-ai/src/
fine_tune/**`, `crates/jammi-ai/tests/it/{pinned_source_gate,training_set,graph_finetune,fine_tune}.rs`,
`crates/jammi-db/src/catalog/result_repo.rs`, `crates/jammi-db/tests/it/materialization.rs`,
`ci/scripts/no_consumer_names_allowlist.txt`, and prose under `docs/`. A1-pull's `P6`/`uses:` rule went
through two earlier, superseded shapes on the same branch — a recursive traversal (at `4a9f5be1`), then
a fail-closed shape with no per-name step-scan (at `f104d15a`) — before the closing-audit sequence
narrowed it once more to the shape P7 below states; this contract cites the two earlier heads only where
P7 itself explains what changed.

Owner: **docs-ci** (dispatched to write this contract only; both units' own code is already committed on
their respective branches). This agent worked in a detached worktree, moving from `4a9f5be1` to
`f104d15a` and finally to `288a34c7` as each landed on the same branch (`$SP/wt-PRB1-contract`, `$SP` =
the session scratchpad), and read every U2a construct with `git show 06720d1e:<path>` — U2a's code is
not present in this worktree's file tree, which sits on the A1-pull branch. Every citation below is
tagged `(A1-pull @ 288a34c7)` or `(U2a @ 06720d1e)` by section, except where P7 itself cites an earlier,
superseded head by name.

## Scope of PR-B1

**U7b-A1-pull** — a paid, opt-in-only 2-GPU CI leg (`.github/workflows/gpu-gang.yml`) that: fails the
leg (never a warning) when its post-run artifact pull fails; keeps the NCCL rendezvous id off every
committed-artifact path; exports the two `JAMMI_REQUIRE_*` require-gate variables the leg's own test
process reads; carries no automated trigger of any kind for the window this merge opens (the `schedule:`
cron is deleted until a sibling unit re-adds it, corrected, beside the writer that proves it); and reads
every GitHub Actions workflow file in this repository through one shared, PyYAML-backed loader that
refuses — by name — every shape it cannot examine soundly.

**U2a** — the fine-tune training-set materialisation path: the tabular arm's own `fine_tune/` surface is
asserted, at the source-text level, to bind no DataFusion session/catalog name and to embed no DDL
statement anywhere in its own tree, and a second, wider literal-occurrence scan reviews every such
binder or DDL literal across both `crates/jammi-ai/src` and `crates/jammi-db/src` (replacing an AST
call-graph gate that was itself excised after execution found call shapes and DDL positions it could not
see); the graph fine-tune arm's own attempt to materialise its sampled pairs through a shared
`SessionContext` was excised back to main's in-memory sampling after two independently-designed guard
mechanisms were each falsified by execution against a real session (issue #538 tracks the rebuild with a
`RecordBatch` source); `ResultTableKind::ALL` is anchored to the compiler's own exhaustiveness check
rather than merely claimed complete.

---

## U7b-A1-pull (A1-pull @ f104d15a)

### P1 — a leg that cannot retrieve its own evidence has proven nothing reviewable

**Property** (quantified over every exit of the pod-tier gang leg): the leg's final return code
reflects a failed post-run artifact pull exactly the way it already reflects the suite's own `rc` —
never a second, independent, best-effort exit path. The pull sits in `ci/scripts/runpod_gpu_gang.sh`'s
top-level post-run block (there is no enclosing function; the leg is a single linear script), executed
after `rc=$?` receives `rp_gang_verdict`'s own verdict and before the exit trap tears the pod down:

```
$ grep -n 'rsync -az' ci/scripts/runpod_gpu_gang.sh
347:  if rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
```

reading, in full, at that anchor (verified by direct read):

```
  if rsync -az -e "ssh ${RP_SSHO[*]} -p ${RP_PORT}" \
    "root@${RP_HOST}:${GANG_REMOTE_ARTIFACT_DIR}/" "${GANG_ARTIFACT_DIR}/"; then
    echo "=== pulled the gang artifact -> ${GANG_ARTIFACT_DIR} ==="
  else
    pull_rc=$?
    echo "::error::gang artifact pull failed (rsync rc=${pull_rc}) -- ..." >&2
    [ "$rc" -eq 0 ] && rc="$pull_rc"
  fi
```

The `[ "$rc" -eq 0 ] && rc="$pull_rc"` line is the join: a pull failure overwrites `rc` only when the
suite itself was already green, and never masks a suite failure that preceded it. A pull failure never
silently downgrades to a warning with `rc` left at `0`.

**Oracle:** `ci/scripts/test_gpu_gang_lane.sh`'s G5 group of cases (`G5: extracted the artifact-
retrieval block from the driver`; `G5: a clean pull with a clean artifact stays rc=0`; `G5: a FAILED
pull (rsync rc=17) with rc=0 so far joins the leg's own rc -> 17, never a silent warning`; `G5: a failed
pull on an ALREADY-failing leg (rc=5) reports the pull failure but never overwrites the leg's own rc`).
**Mutation:** restoring the pre-fix warning-only form (`rsync … || echo "::warning::…"` with no `rc`
join) is exactly the shape the rc=17 case is built to kill — a warning never changes `$?`, so that case's
assertion of `rc=17` cannot pass against it.

### P2 — the NCCL rendezvous id never rides a path this driver commits

**Property:** the id file (not yet minted by any code on this branch — a downstream unit mints it) is
documented, in the driver's own committed text, to live outside the directory this driver pulls back and
a human later commits from:

```
$ grep -n 'rides OUTSIDE' ci/scripts/runpod_gpu_gang.sh
364:# rides OUTSIDE ${GANG_REMOTE_ARTIFACT_DIR}/${GANG_ARTIFACT_DIR} -- never
```

with the full comment block, read in place:

```
# The NCCL id (128 opaque bytes minted by rank 0) crosses hosts ONLY
# hex-encoded; it must never reach a committed artifact or a CI log -- it
# is the capability to join this gang, not evidence of one; the driver
# never sees the id. This lane mints/ships no id today. The id file's own
# committed contract, fixed here BEFORE that mechanism exists, is that it
# rides OUTSIDE ${GANG_REMOTE_ARTIFACT_DIR}/${GANG_ARTIFACT_DIR} -- never
# inside the directory this driver pulls back and a human later commits.
# The scan that backstops this contract against a future mistake ships
# with the cluster leg (docs/plans/67-distributed-training/UNITS.md § U7b
# acceptance (id-secrecy)), beside the id-ship crossing it protects.
```

The scan that would enforce this property against a real carrier set (the pulled directory, the log, any
staging path) is not part of this unit — it was excised after a block on its own soundness (grep's exit
lattice masked by a swallowed non-zero exit; in-tree symlinks and archive carriers skipped silently) —
and is scheduled, by citation only, on the plan document's own U7b section:

```
$ grep -n 'id-secrecy' docs/plans/67-distributed-training/UNITS.md
218:- **acceptance (id-secrecy)**: the NCCL id (the 128-byte secret that crosses hosts hex-encoded)
```

The driver names that obligation, never reproduces it, at three sites (each reading
identical text): the header comment, the artifact-retrieval comment quoted above, and a third comment
inside the exit-code block.

### P3 — the require-gate exports are present, exactly two, and the sentences describing them are true

**Property:** the leg's remote heredoc exports both atoms:

```
$ grep -n 'JAMMI_REQUIRE_CUDA' ci/scripts/runpod_gpu_gang.sh
248:# than skip: JAMMI_REQUIRE_CUDA_GANG is exported by no other driver in this
250:# false green with no gang ever proven. The plain JAMMI_REQUIRE_CUDA half of
257:export JAMMI_REQUIRE_CUDA=1
258:export JAMMI_REQUIRE_CUDA_GANG=1
```

and every in-tree sentence describing either variable is true in the present tense. Both sites live on
real Rust items whose own doc comments this agent read in full at the head cited:
`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs::serial_cuda_device_or_require` and
`crates/jammi-ai/tests/gpu_capability/gang_nccl.rs::second_cuda_device_or_require` — both doc comments
name `ci/scripts/runpod_gpu_gang.sh`'s remote heredoc as the exporter, in the present tense ("the gang
pod lane's remote heredoc … exports `JAMMI_REQUIRE_CUDA_GANG=1`"), with no "does not exist yet" or "will
export" hedge (that hedge was present at this unit's own base and is the specific defect this property
closes).

**Oracle:** `ci/scripts/test_gpu_gang_lane.sh`'s G6 group (`G6: the expanded remote heredoc exports both
JAMMI_REQUIRE_CUDA=1 and JAMMI_REQUIRE_CUDA_GANG=1`; `G6: exactly two JAMMI_REQUIRE_CUDA* export lines
(removing either is the mutation this case catches)`). **Mutation:** deleting either export line is
exactly the shape the second G6 case is built to catch.

### P4 — `PROVE_EXPECT_SHA` pins the code tree; the image's own tag pins the image; the two are never the same value

**Property:** the leg clones the code tree by branch, never by a bare sha:

```
$ grep -n 'git clone --depth 1\|PROVE_SHA=' ci/scripts/runpod_gpu_gang.sh
287:git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
289:echo "PROVE_SHA=\$(git rev-parse HEAD)"
```

`git clone --depth 1 -b "${GIT_REF}"` cannot take a bare sha as `${GIT_REF}` and still resolve, so the
echoed `PROVE_SHA` is always `${GIT_REF}`'s own head at clone time — never a separately-built image's
own commit. This is documented, not newly coded: the driver's own header names the refusal exit code
this ambient mismatch trips:

```
$ grep -n 'wrong tree' ci/scripts/runpod_gpu_gang.sh
82:# wrong tree (the pod's own PROVE_SHA disagreed with PROVE_EXPECT_SHA); 97 =
```

No new driver code implements the read/refusal itself — `ci/scripts/runpod_lib.sh::rp_run_remote_
watched`'s ambient `PROVE_EXPECT_SHA` read and its refusal predate this unit; this unit's own obligation
is documented accuracy about which sha it pins, discharged by the header text quoted above.

### P5 — no automated trigger of any kind fires the paid leg between this merge and the unit that re-adds one

**Property:** `.github/workflows/gpu-gang.yml` carries no schedule trigger of any kind at this head:

```
$ grep -n 'schedule\|cron' .github/workflows/gpu-gang.yml
(no output -- no match)
```

The block that existed at this unit's own base (`9af28de0`) — a daily cron and its preceding comment,
including a materially false claim about the reaper's own period — is deleted, together with every
prose site that named it, in this unit's own commit set. Only the `run-gang` PR label (a human, per-PR
opt-in) or a manual `workflow_dispatch` (also a human act) can run the leg in this window; neither is
automated. This is a temporary, intentionally incomplete state, not an oversight: the cron is
re-introduced, corrected, by a sibling unit not part of PR-B1, in the same commit as the writer that
makes a real run of it pass.

**Oracle (a property of this commit's own diff, not a permanent invariant):** `ci/scripts/
test_gpu_gang_lane.sh`'s G7 case (`G7: NO schedule: key exists anywhere in the committed gpu-gang.yml`).
This assertion is itself scheduled for deletion the moment the sibling unit's schedule-re-add commit
lands — a permanent "no schedule key" assertion would then reject the correct end state — which the
suite's own header states in prose (this agent read the full G7 header comment block naming this
scoping; it is not reproduced verbatim here to avoid the "reproduces rather than names" defect this
same header itself warns against for its own upstream deletion).

### P6 — every GitHub Actions workflow reader in this repository resolves from one parsed document, and refuses, by name, what it cannot examine soundly

**Property** (quantified over every `on:` trigger read, every `jobs:` block read, and every `uses:`
target resolved by any gate under `ci/scripts/`): the value comes from a `yaml.safe_load`-equivalent
parse under a loader that refuses a document reusing a mapping key at the same level
(`ci/scripts/check_execution_surface_reachability.py::_NoDuplicateKeysSafeLoader`) and refuses any YAML
anchor, alias (including a merge key), or explicit tag before trusting the parse at all
(`ci/scripts/check_execution_surface_reachability.py::_assert_no_github_incompatible_yaml`) — GitHub
Actions' own parser rejects all three; a reader that resolved them anyway would parse documents GitHub
itself refuses to run. `ci/scripts/check_execution_surface_reachability.py::load_workflow_text` and
`::load_workflow_from_path` are the one composition point; every read error — a missing PyYAML install,
a syntax error, a non-mapping top level, a duplicate key, an anchor/alias/tag, a boolean/string
`on`-key collision, an unreadable file for any euid — raises the single named
`ci/scripts/check_execution_surface_reachability.py::WorkflowLoadError`, never a silent `{}` or `[]`
standing in for "nothing found here."

Two other gates import this one reader rather than keeping their own:
`ci/scripts/check_gpu_prove_once.py::main` imports the module as `exec_mod` and calls
`exec_mod.require_pyyaml_or_exit("gpu-prove-once", exit_code=3)` before any real work, and
`ci/scripts/check_lint_surface_closure.py::lanes_from_workflows` calls `exec_mod.scan_workflows(...)`
and propagates that function's own findings list rather than discarding it — its own doc comment states
why: "a workflow this reader cannot parse might be the one that actually hosts … a required lane."

```
$ grep -n 'import check_execution_surface_reachability\|exec_mod\.' ci/scripts/check_gpu_prove_once.py | head -3
233:import check_execution_surface_reachability as exec_mod  # noqa: E402
2360:    prereq_rc = exec_mod.require_pyyaml_or_exit("gpu-prove-once", exit_code=3)
2361:    if prereq_rc is not None:
```

### P7 — a merge-path job's `uses:` is judged fail-closed, with no traversal into a delegate's own jobs, and the two reviewed exemptions get their own compensating step-scan

**This property went through two earlier, superseded shapes on the same branch before reaching the one
that ships.** At `4a9f5be1` it was a recursive traversal (`job_invokes_publish_primitive_recursive`
walking into every reusable a job-level `uses:` named, via a `_traverse_reusable` helper, a depth bound,
and a per-path memo) — deleted after a cross-repo `uses:` call produced no finding. At `f104d15a` the
replacement fail-closed shape treated `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` as a bare name exemption
with no further examination of what those files' own jobs might do. The unit's own closing-audit
sequence narrowed this once more: a `PROMOTION_TABLE` row's `gate_job` exemption is now valid ONLY when
that job's own parsed job-level `uses:` resolves EXACTLY to the proof-required workflow (any other
value on that same job, including a non-string one, is still a finding), and BOTH exempted-by-name
classes now get a compensating, non-recursive scan of every job directly inside them. This agent
re-detached to the head that carries this shape, `288a34c7`, and re-verified every construct below by
direct read, confirming the earlier heads' now-deleted names are in fact absent:

```
$ git log --oneline -3
288a34c7 ci(gates): narrow the gate_job exemption to its own real uses:, scan name-exempted files directly, restore the whole-step primitive domain
f104d15a docs(ci): cite the filed traversal-rebuild issue by number in P6's docstring
a895b148 ci(gates): P6's uses: rule ships fail-closed with no traversal; push: and P3 read the parsed steps; branch-authored bookkeeping markers deleted
$ grep -n 'job_invokes_publish_primitive_recursive\|_traverse_reusable\|_MAX_USES_DEPTH\|_push_value_is_promoting\|_PUSH_VALUE_RE' ci/scripts/check_gpu_prove_once.py
(no output -- no match)
```

**Property (fail-closed, no traversal; quantified over every merge-path job):** a job is a "promotion
job" for P6's purposes when either one of its own steps invokes a listed primitive directly, or it
carries a job-level `uses:` delegating to ANY other workflow at all — local, cross-repo, a dangling
target, quoted, a `+`-bearing filename, all alike, since the rule reads the raw parsed scalar's mere
presence, never its resolved identity; a job-level `uses:` value that is present but not a string is
its own named finding. Either way the job must be listed in `PROMOTION_TABLE` (any row's
`promoting_job`) or the gate fails by name, with exactly TWO narrow, hand-reviewed exceptions, neither a
traversal: (1) a table row's own `gate_job` is exempt only when its OWN parsed job-level `uses:`
resolves exactly to the proof-required workflow; (2) a local target named in
`REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` is exempt by name. `check_gpu_prove_once.py
::check_p6_discovery`'s own doc comment states this precisely:

```
def check_p6_discovery(workflow_texts: dict[str, str]) -> list[str]:
    """PROPERTY (fail-closed, no traversal into a delegate's own jobs): a
    merge-path job is a "promotion job" for P6's purposes when EITHER (a)
    one of its own steps invokes a listed primitive directly
    (`job_invokes_publish_primitive`), OR (b) it carries a job-level
    `uses:` delegating to ANY other workflow at all -- local, cross-repo,
    a dangling target, quoted, a `+`-bearing filename, all alike, since
    the raw parsed scalar's mere PRESENCE is what this rule reads, never
    its resolved identity; a job-level `uses:` value that is PRESENT but
    NOT a string is its own named finding (cannot examine at all). Either
    way, the job must be listed in `PROMOTION_TABLE` (any row's
    `promoting_job`) or this gate fails by name. A job-level delegation is
    judged WITHOUT ever opening the delegate's own text: this rule does
    not know or care whether the delegate itself promotes anything, is
    examinable, or even exists -- ANY delegation is presumed promoting
    until a human reviews it, with TWO narrow, hand-reviewed exceptions,
    NEITHER of which is a traversal: (1) a `PROMOTION_TABLE` row's own
    `gate_job` is exempt ONLY when its OWN parsed job-level `uses:`
    scalar resolves EXACTLY to `PROOF_REQUIRED_WORKFLOW` (either
    extension spelling) -- any other value on that same job, including a
    non-string one, is still a finding; (2) a LOCAL target in
    `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` is exempt by NAME. Both
    exempted-by-name classes (`REVIEWED_NONPUBLISHING_LOCAL_REUSABLES`
    and `PROOF_REQUIRED_WORKFLOW` itself) get a compensating, NON-
    recursive top-level examination below: every job INSIDE the exempted
    file is scanned by the step-level publish-primitive rule directly ...
    the exemption is a reviewed NAME plus that direct scan, never an
    examination of what the exempted file's OWN jobs might themselves
    delegate to (rebuilding a real per-delegate traversal is tracked as
    https://github.com/f-inverse/jammi-ai/issues/561). ..."""
```

The step-scan target set is resolved once per run by `check_gpu_prove_once.py
::_resolved_exempt_step_scan_names`:

```
def _resolved_exempt_step_scan_names(workflow_texts: dict[str, str]) -> set[str]:
    """The discovered-on-disk spellings (either extension) of every
    workflow this module scans directly by NAME instead of resolving via
    a table row: `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` and
    `PROOF_REQUIRED_WORKFLOW` (`_gpu-proof-required.yml`)."""
```

— three files: `_summary.yml`, `_pypi-server.yml` (both in `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES`,
unchanged from the earlier shape) and `_gpu-proof-required.yml` (`PROOF_REQUIRED_WORKFLOW`, newly
step-scanned at this shape). Every job inside these three files is examined directly, via
`check_gpu_prove_once.py::job_invokes_publish_primitive` — the SAME whole-job reader used everywhere
else in this gate:

```
def job_invokes_publish_primitive(job_node: dict) -> tuple[str | None, str | None]:
    """(primitive, error). `primitive` is the matched display name for the
    first (in order) step that invokes one -- a job whose steps invoke ANY
    listed primitive is a "promotion job" for P6's purposes: it must be
    listed in `PROMOTION_TABLE` (any row, any gate_kind) or this gate
    fails by name. Reads every step from the PARSED document via
    `_step_invokes_publish_primitive`. `error` is set (primitive always
    `None`) the moment any step entry is not itself a mapping ... A
    job-level `uses:` (this job itself delegating to another workflow,
    carrying no `steps:` of its own) is a SEPARATE, fail-closed rule
    `check_p6_discovery` applies directly -- see its own docstring."""
```

which in turn reads every step's scalars through `check_gpu_prove_once.py::_step_scalar_values` — the
construct that restores the whole-step primitive domain a narrower interim shape had lost:

```
def _step_scalar_values(step_node: dict) -> list[str]:
    """Every string scalar this step's own body can carry a publishing
    marker in -- `run`, `uses`, every value under `with:`, every value
    under `env:` -- the whole-step domain this matcher is held to,
    including `with.command`/`with.script`/`with.args`/`with.entrypoint`/
    `env`-carried markers, never `run`/`uses` alone.
    `name`/`id`/`if` are deliberately EXCLUDED: identifiers and
    conditions, never invocation content. A non-string `with:`/`env:`
    value (a bool, a number) cannot itself match a substring pattern and
    contributes nothing here."""
```

The gate_job exemption's own resolver, `check_gpu_prove_once.py::_local_reusable_workflow_target`, reads
the job-level `uses:` from the parsed mapping directly (never a text regex, so a quoted value or a
`+`-bearing filename are both read correctly):

```
def _local_reusable_workflow_target(job_node: dict) -> str | None:
    """The job-level `uses: ./.github/workflows/<X>.yml` target THIS job
    delegates to (never a step-level action `uses:`, which lives under a
    DIFFERENT key, `steps:`, and is `job_invokes_publish_primitive`'s own
    concern) -- read directly from the job's own PARSED mapping ... `None`
    when this job has no job-level `uses:` at all, or that value does not
    name a LOCAL workflow."""
```

and the name-exemption check, `check_gpu_prove_once.py::_job_level_uses_is_reviewed_nonpublishing`, is
confirmed still called from `check_p6_discovery` by grep (its own docstring's closing parenthetical, a
leftover from an earlier shape, claims `check_p6_discovery` "does not need this function at all" — this
agent verified that claim is stale: `grep -n '_job_level_uses_is_reviewed_nonpublishing(' ci/scripts/
check_gpu_prove_once.py` shows the definition AND one call site inside `check_p6_discovery`'s own body):

```
def _job_level_uses_is_reviewed_nonpublishing(job_node: dict) -> bool:
    """`True` when this job's own job-level `uses:` names a LOCAL
    reusable in `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` (either extension
    spelling) -- never true for a cross-repo, dangling, or otherwise
    unreviewed target, which stays fail-closed."""
```

`check_gpu_prove_once.py::_other_publishing_steps` is P3's own sibling check on a step-gated row — every
step in that row's job OTHER than the one named step the row pins, read the same way `job_invokes_
publish_primitive` reads a whole job, so a second, ungated publishing step in the same job cannot sail
through unseen:

```
def _other_publishing_steps(
    job_node: dict, gated_step_name: str
) -> tuple[list[tuple[str, str]], str | None]:
    """(`[(step_name, matched_primitive), ...]`, error) for every step in
    this job OTHER than `gated_step_name` -- a step-gated row (P3) only
    ever pins the NAMED step's `if:`; a second, ungated publishing step in
    the SAME job used to sail through unseen. Reads `steps:` from the
    PARSED document via `_step_invokes_publish_primitive`, the SAME
    reader `job_invokes_publish_primitive` uses ... `gated_step_name`
    exclusion is by DISPLAY NAME ONLY -- a second, ungated step sharing
    the SAME `name:` as the genuinely gated one is excluded here too and
    stays invisible, the same identification gap `find_step_if_by_name`
    carries. See https://github.com/f-inverse/jammi-ai/issues/564."""
```

`check_gpu_prove_once.py::check_promotion_table` is P3's own gate over the same table `check_p6_discovery`
reads:

```
def check_promotion_table(workflow_texts: dict[str, str], manifest: dict) -> list[str]:
```

— the subset check between the release manifest's CUDA lanes and the table's rows, and the structural
`if:` proof required of every deliberately-ungated `gate_kind="none"` row (a `gate_kind="none"` row's
promoting job must carry the exact conjunct excluding a release-tag ref, never merely "no `refs/tags/`
substring") — confirmed present and substantial by direct read, not reproduced in full here beyond its
own def line and the two rules named above.

**Real-corpus coverage, executed by this agent against the actual `.github/workflows/*.yml` tree at
`288a34c7`** (not asserted from prose): scanning every workflow's every job for a job-level `uses:`
finds exactly 14 delegations. Of those, 12 point at the three step-scanned exempt files (8 at
`_gpu-proof-required.yml`, 3 at `_pypi-server.yml`, 1 at `_summary.yml`) and are covered by the direct
step-scan above. The remaining 2 — `image.yml`'s `build` job and `image-cuda.yml`'s `build` job, both
targeting `_ci-base-image.yml` — are neither traversed, nor step-scanned by name (`_ci-base-image.yml`
is in neither `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` nor equal to `PROOF_REQUIRED_WORKFLOW`); they are
simply LISTED as reviewed `gate_kind="none"` rows in `PROMOTION_TABLE`, confirmed by direct read of the
table:

```
$ python3 -c "
import sys; sys.path.insert(0, 'ci/scripts')
import check_gpu_prove_once as m
for key, row in m.PROMOTION_TABLE.items():
    if row.workflow in ('image.yml', 'image-cuda.yml'):
        print(key, row.workflow, row.promoting_job, row.gate_kind, row.gate_job)
"
ci-image-cpu image.yml build none None
ci-image-cuda image-cuda.yml build none None
```

so a publishing primitive planted directly inside `_ci-base-image.yml`'s own job body is invisible to
this gate by construction — no traversal opens that file's own text, and no by-name step-scan covers it
either. This is exactly the residual issue #561 tracks (see below), stated here with the real corpus
count rather than a hypothetical.

**The reusable-workflow traversal's rebuild is filed at issue #561** (title verified below); the four
residuals from the last closing audit, all Stands filed rather than fixed in this shape, are listed
together in "Residuals recorded UNCOVERED" below with the construct each one names.

### PyYAML as a declared, checked prerequisite

**Property:** every gate depending on the shared loader checks for PyYAML's presence before dispatching
`--self-test` or any real run, emitting one distinct "gate prerequisite missing: PyYAML" message and a
distinct exit code — never a bare `import yaml` stack trace, and never treated as a finding.
`ci/scripts/check_execution_surface_reachability.py::require_pyyaml_or_exit` is the one definition,
called from that module's own `main` (via its `_pyyaml_prerequisite_rc` wrapper) and from
`ci/scripts/check_gpu_prove_once.py::main`. The dependency is installed for two different execution
surfaces, each with its own reason stated in its own comment:

```
$ sed -n '20,48p' .docker/ci.Dockerfile
```
```
# PyYAML: a declared prerequisite of `ci/scripts/check_execution_surface_
# reachability.py`'s shared YAML-backed workflow loader (and every gate
# that imports it: check_gpu_prove_once.py, check_lint_surface_closure.py).
# ...
RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir 'PyYAML==6.*'
```

and, as an interim measure for the window before that rebuilt image is the one every job pulls, two
workflow-level steps:

```
$ grep -n 'Install PyYAML' .github/workflows/ci.yml
632:      - name: Install PyYAML (lint-surface-closure needs it inside the container)
2425:      - name: Install PyYAML (execution-surface-reachability / gpu-prove-once / gpu-gang-lane legs only)
```

`.docker/ci.Dockerfile`'s own surrounding comment states why `ensurepip`'s ordinary bundled-wheel path
fails on this distro's Python image and what the shown form resolves to instead (read in full by this
agent; not reproduced here beyond the excerpt above).

---

## U2a (U2a @ 06720d1e)

### P8 — the graph fine-tune arm is main's, unchanged; no per-call session/catalog binding lives anywhere under `fine_tune/`

**Property:** `run_spec`'s `GraphFineTune` arm and `reconstruct_graph_loader` sample in memory and build
a loader exactly as `origin/main` does — no `TrainingSet` table is written for a graph job, nothing is
registered on the shared `SessionContext`, and no `jammi_sampled_pairs` relation exists anywhere in
`crates/jammi-ai`. This is the end state of an excision, not an original design choice: two
independently-designed guard mechanisms — a `RegisteredPairs` occupancy-refusal type, then a per-job
UNIQUE relation-name scheme with a scope-guard deregistration — were each falsified by execution against
a REAL `InferenceSession` (never only a bare `SessionContext::new()` fixture, whose own green oracle had
proven only DataFusion's in-memory catalog's own refusal behaviour, not production's real schema
provider): a second overlapping registration silently rebinds the first job's relation on production's
real schema provider, and a reclaim-driven second materialisation of the SAME job id displaces the first
attempt's binding with no detectable signal.

**The producer's SQL-only shape — the reason the excision took this final, more severe form — is not
citable against `06720d1e` by this agent**, because the graph-arm producer code it would describe was
itself the thing removed by the excision; there is nothing left at that head to grep. It IS, however,
stated in the filing that records the excision, **issue #538**, whose body this agent read in full via
`gh issue view 538 --json title,body`:

```
$ gh issue view 538 --json title
{"title":"fine-tune: graph arm through the materialised TrainingSet table (excised from #500 U2a)"}
```

quoting the exact sentence from that issue's own body that names the producer construct:

```
The producer (`ResultStore::materialize_training_set`, `TrainingSetSpec.source_sql`) executes SQL
only, so the nameless in-memory batch source the seven sibling producers use
(`ctx.read_table(provider)`) is not available to it.
```

This is a citation of the FILED ISSUE's own text, not an independent construct-read by this agent
against any commit — the issue is the only place this contract can point to for the reason, since the
code itself is gone. (**Correction of the coordinator's suggested source:** issue #554, which this
agent also read in full, does NOT discuss the graph-arm producer or its SQL-only shape at all — its
entire body concerns the literal-occurrence gate's own occurrence-counting/split-literal/`include_str!`
residuals (P10 below). The producer citation belongs to #538, not #554; this agent does not cite #554
for P8.)

The rebuild — the producer taking a `RecordBatch` source as a first-class input beside SQL, with its own
anchor rule for an in-memory source and a definition hash that folds no run-scoped token — is scheduled
as this same issue, #538, per its own body: "a `RecordBatch` source (not SQL) with a stated anchor rule
for an in-memory input (the graph arm's anchors are the node and edge sources), a definition hash that
does not fold a run-scoped token, and an overlap oracle whose two calls share ONE job id."

### P9 — no file under `crates/jammi-ai/src/fine_tune/**` itself spells a session-binding verb or a DDL literal

**Property** (quantified over every `.rs` file tracked under `crates/jammi-ai/src/fine_tune/`): a direct
text-level scan finds zero occurrences of any DataFusion session/catalog (de)registration verb and zero
DDL-statement-shaped string literals in that tree — the cheap, zero-tolerance DIRECT layer, a subset of
the wider scan (P10) below.

**Oracle:** `crates/jammi-ai/tests/it/pinned_source_gate.rs::no_session_registration_under_fine_tune`
(retained, per the excision's own stated boundary: this agent read the gate file's own section comment,
which states in prose that this check and its falsifications "stay here, cheap and text-based, as the
first line of defense") and `crates/jammi-ai/tests/it/pinned_source_gate.rs::
fine_tune_ddl_relation_binding_hits`, whose falsification battery includes a synthetic `CREATE VIEW`
DDL-string shape (must be flagged), a comment-embedded DDL string (must NOT be flagged), and a
directory-scoping control (must not see a DDL literal outside `fine_tune/`) — all three confirmed
present as named test functions in the file this agent read via `git show 06720d1e:crates/jammi-ai/
tests/it/pinned_source_gate.rs`.

### P10 — every registration-verb call or DDL-shaped literal anywhere under BOTH crates' `src` trees is either reviewed by name or flagged

**Property** (quantified over every physical line under `crates/jammi-db/src` and `crates/jammi-ai/src`
— `crates/jammi-ai/tests/it/pinned_source_gate.rs::SURFACE_DIRS`): a line matching one of the 24
`register_*`/`deregister_*` call-site patterns
(`crates/jammi-ai/tests/it/pinned_source_gate.rs::PAIRED_REGISTRATION_VERBS` /
`::UNPAIRED_REGISTRATION_VERBS`) or the DDL-statement shape is attributed to its enclosing function and
keyed by `(file, function, ordinal)` — never a bare line number, which drifts under an unrelated edit
above it (`crates/jammi-ai/tests/it/pinned_source_gate.rs::assign_ordinals` is the ordinal-assignment
function whose own doc comment states this reasoning). Every such key is either present on a reviewed
allowlist (`crates/jammi-ai/tests/it/pinned_source_gate.rs::ReviewedRegistrationSite`, each entry
carrying a non-trivial `property` string) or the enumerating test fails, naming the site.

This gate REPLACES a `syn`-based AST call-graph gate that was itself deleted after execution found call
shapes with no edge (a fn-pointer argument, a `.map(Self::f)` call, a call inside a macro invocation, a
`tokio::select!` arm, a fn-pointer struct field) and DDL positions invisible to both layers (a
module-level `const`, a DDL string split across two `format!`/`concat!` arguments, a
`concat!`-assembled literal, an `include_str!` target) — a soundness defect in what any finite set of
AST node-kind handlers can promise to cover exhaustively, per that section's own comment, which tracks
the reachability question itself at **issue #549**:

```
$ gh issue view 549 --json title
{"title":"gate: sound fine_tune→DataFusion-binder reachability (call graph over every edge shape, DDL in every literal position, universe = dependency closure)"}
```

**Oracles:** `crates/jammi-ai/tests/it/pinned_source_gate.rs::registration_verb_occurrences_are_
all_reviewed`, `::ddl_literal_occurrences_are_all_reviewed`, `::every_reviewed_registration_site_states_
its_property` (asserts every reviewed entry's `property` field is non-decorative prose, over 20
characters). **Falsification controls, each with the mutation it survives:**
`::falsification_new_verb_occurrence_in_a_new_file_is_flagged` (plants a `ctx.register_table(...)` call
in a synthetic new file and asserts the scan finds it AND that it is not already on the reviewed list,
so this control cannot be vacuous) and `::falsification_removing_a_reviewed_entry_leaves_its_site_
unreviewed` (reproduces the real reviewed site `crates/jammi-ai/src/query/content_hash_udf.rs::
register_content_hash_udf` as a synthetic fixture string and asserts an EMPTY allowlist reports it
unreviewed — i.e. removing the real entry from the real allowlist would turn the real test RED). This
agent confirmed all four function names exist in the file at `06720d1e` by direct `git show | grep -n`.

**Stated residual, honestly, not as a solved problem:** this scan counts SITES, never per-site
OCCURRENCES (a second call planted inside an already-reviewed function raises no new hit); it is
strictly line-based (a DDL literal split across two `concat!`/`format!` arguments on separate lines, or
pulled in through `include_str!`, is invisible to it — the same two shapes the deleted AST gate also
missed); it is scoped to exactly `SURFACE_DIRS` (a verb or literal under `tests/it/` in either crate, or
in a third crate, is outside its universe — the gate's own comment names five real `.register_table(`
calls in `crates/jammi-db/tests/it/materialization.rs` as the concrete example); and its own masking
step, `crates/jammi-ai/tests/it/pinned_source_gate.rs::mask_comments_only`, desyncs on a raw string as
the gate ships. That function's own doc comment states an executed measurement of the desync (this
agent read it verbatim, quoted here inside a fence rather than retyped as a bare claim):

```
Measured with this exact function, compiled verbatim (not a
transcription), over both `src` trees at this head: 22 code lines get
blanked as if they were comments and 12 real `//` comment lines are left
completely unblanked, across `crates/jammi-db/src/storage/config.rs` (its two single-line raw strings),
`config/tests.rs`, `config/secret.rs`, and `sql/ident.rs` -- the complete
set.
```

All five gaps (per-site counts, split literals, `include_str!` targets, anything outside `SURFACE_DIRS`
including `tests/it/`, the masking step's raw-string desync) are named, together, as the open universe
this gate does not close, tracked at **issue #554**:

```
$ gh issue view 554 --json title
{"title":"U2a literal-occurrence gate: bind site keys to occurrence counts; cover split literals and include_str! targets"}
```

### P11 — a reviewed, mechanism-only allowlist row exists for the falsification fixture's own reproduced identifier

**Property:** `ci/scripts/no_consumer_names_allowlist.txt`'s governance-verb tripwire flags
`register_content_hash_udf` inside `pinned_source_gate.rs` because the falsification fixture above (P10)
deliberately reproduces the real reviewed site's source text as a string literal; the added row is
scoped to the exact `(identifier, declaring_path)` pair — its second field is
`crates/jammi-ai/tests/it/pinned_source_gate.rs`, distinct from the pre-existing row's
`crates/jammi-ai/src/query/content_hash_udf.rs` — and its own comment states the ruling was MECHANISM,
not governance, citing issue #554 for the ruling (this agent read the full commit `b9c07ea7` on
`feat/500-B-U2a` directly). This file is gate data under `swarm.yml`'s human-amend-only glob
(`SWARM_GATE_TOUCHED`); the row is flagged for human review at PR-B1's merge, not landed as a silent
edit.

### P12 — `ResultTableKind::ALL` is anchored to the compiler's own exhaustiveness check, and its own doc states exactly what that anchor does and does not guarantee

**Property:** `crates/jammi-db/src/catalog/result_repo.rs::ResultTableKind::ALL` is produced by
`crates/jammi-db/src/catalog/result_repo.rs::ResultTableKind::all`, a `const fn` whose body is one
`match` over `Self::Model` with every variant named in a single arm pattern joined by `|` — appending a
variant to the enum without adding it to that pattern is a compile error (E0004, "non-exhaustive
patterns"). This agent read both items in full via `git show 06720d1e:crates/jammi-db/src/catalog/
result_repo.rs`:

```
pub const ALL: [Self; 4] = Self::all();

const fn all() -> [Self; 4] {
    match Self::Model {
        Self::Model | Self::NeighborGraph | Self::AsofJoin | Self::TrainingSet => [
            Self::Model,
            Self::NeighborGraph,
            Self::AsofJoin,
            Self::TrainingSet,
        ],
    }
}
```

The doc comment states the boundary honestly rather than overclaiming: "E0004 binds the pattern, not the
`[Self; 4]` array literal — so a variant named in the pattern but omitted from the array still
compiles. `ALL` therefore names every variant the pattern names, not necessarily every variant of
`Self`" — citing a separate, pre-existing tracking issue (#550, named inline in that doc comment, not one
of the four issue numbers this contract was asked to verify and not independently re-checked via
`gh issue view` in this session) for closing that gap generally. This replaces a tautological oracle
(both sides of which were derived from `ALL` itself, proving nothing) with a compiler-enforced one.

---

## Gate files a human must review at PR-B1's merge

Both units edit files inside `swarm.yml`'s human-amend-only glob; `SWARM_GATE_TOUCHED` fails by design
and this is what the reviewer checks:

- `ci/scripts/check_execution_surface_reachability.py` — the shared PyYAML-backed loader
  (`_NoDuplicateKeysSafeLoader`, `_assert_no_github_incompatible_yaml`, `load_workflow_text`,
  `load_workflow_from_path`, `require_pyyaml_or_exit`), the `on:`/`jobs:` readers.
- `ci/scripts/check_gpu_prove_once.py` — the fail-closed job-level `uses:` check (P7 above:
  `check_p6_discovery`, `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES`, `_resolved_exempt_step_scan_names`,
  `_job_level_uses_is_reviewed_nonpublishing`, `_local_reusable_workflow_target`,
  `job_invokes_publish_primitive`, `_step_scalar_values`, `_step_push_is_promoting`,
  `_other_publishing_steps`, `_step_invokes_publish_primitive`, `check_promotion_table`; the recursive
  traversal this contract found present at the earlier `4a9f5be1` head is deleted, its rebuild filed at
  issue #561 and cited by number in this module's own docstring), its own PyYAML prerequisite call.
- `ci/scripts/check_lint_surface_closure.py` — `lanes_from_workflows` now consumes
  `check_execution_surface_reachability.scan_workflows`'s findings rather than discarding them.
- `ci/scripts/no_consumer_names_allowlist.txt` — the new `register_content_hash_udf` row scoped to
  `crates/jammi-ai/tests/it/pinned_source_gate.rs` (P11 above).
- `.docker/ci.Dockerfile` — the PyYAML `ensurepip`/`pip install` bootstrap.
- `.github/workflows/ci.yml` — the two interim PyYAML install steps ("Install PyYAML (lint-surface-
  closure needs it inside the container)" and "Install PyYAML (execution-surface-reachability /
  gpu-prove-once / gpu-gang-lane legs only)") that cover every job running a loader-dependent gate before
  the CI image itself is rebuilt with PyYAML baked in.

## Residuals recorded UNCOVERED

Four residuals were filed as Stands by the last closing audit on `check_gpu_prove_once.py`'s P6/P3
shape, none fixed in this shape (per that audit's own disposition — each is a filed finding, not a
mechanism this contract can claim closed):

- **[Issue #561](https://github.com/f-inverse/jammi-ai/issues/561)** — verified via `gh issue view 561
  --json title` (quoted under P7 above): P7's fail-closed shape refuses every job-level `uses:` on a
  merge-path job unless that job is listed by name or targets one of the two reviewed non-publishing
  reusables; it does not resolve any OTHER reusable's own reachable jobs at all — a real, examinable,
  non-promoting reusable workflow reached through a listed job's `uses:` is refused the same way an
  unexaminable one is, because no traversal exists to tell them apart. On the real corpus this is
  concretely the `_ci-base-image.yml` case P7 measures (2 of 14 delegations, both reviewed
  `gate_kind="none"` rows, neither traversed nor step-scanned) — a primitive planted inside that file is
  silent. #561 is the rebuild that restores the distinction soundly; `check_p6_discovery`'s own
  docstring cites #561 by URL and states that this fail-closed shape supersedes it until it lands.
- **[Issue #563](https://github.com/f-inverse/jammi-ai/issues/563)** — verified via `gh issue view 563
  --json title`: `{"title":"P6: _step_push_is_promoting inherits PyYAML's YAML-1.1 boolean set (no/off
  read as false), wider than GitHub's"}` — `_step_push_is_promoting`'s fail-closed rule accepts the
  literal `False`/`"false"` as non-promoting, but PyYAML's own YAML-1.1 boolean resolution also maps
  `no`/`off` (and their case variants) to Python `False` before this function ever sees the value, a
  wider "false" set than GitHub Actions itself recognises for a `with:` input.
- **[Issue #564](https://github.com/f-inverse/jammi-ai/issues/564)** — verified via `gh issue view 564
  --json title`: `{"title":"P3: the step-gated row identifies the gated step by display name, so a
  duplicate-named second publishing step is skipped by both readers"}` — `_other_publishing_steps`'s own
  docstring states this exactly: exclusion by `gated_step_name` is by display name only, so a second,
  ungated step sharing the same `name:` as the genuinely gated one is excluded too and stays invisible.
- **[Issue #565](https://github.com/f-inverse/jammi-ai/issues/565)** — verified via `gh issue view 565
  --json title`: `{"title":"P6/P3 publish-primitive domain is step-scalar-only: job-level env:/
  strategy.matrix and non-scalar with:/env: carriers are silent"}` — `_step_scalar_values` covers a
  step's own `run`/`uses`/`with.*`/`env.*` string scalars only; a publishing marker carried in a JOB-level
  `env:` block, a `strategy.matrix` value, or a non-string `with:`/`env:` value is outside every reader
  built on `_step_scalar_values` (`_step_invokes_publish_primitive`, and therefore
  `job_invokes_publish_primitive` and `_other_publishing_steps` too).
- **Issue #549** — verified via `gh issue view 549 --json title` (quoted under P10 above): the INDIRECT
  reachability question the deleted AST call-graph gate existed to answer (an in-tree function OUTSIDE
  `fine_tune/` that itself binds a session/catalog name, reached through some chain of in-tree calls
  `fine_tune/` makes) is open; P10's literal-occurrence gate answers a different, narrower question
  (every occurrence, regardless of reachability) and does not close it.
- **Issue #554** — verified via `gh issue view 554 --json title` (quoted under P10 above): the five gaps
  named under P10 (per-site occurrence counting, split-literal assembly, `include_str!` targets, the
  `SURFACE_DIRS` boundary excluding `tests/it/` and other crates, and the masking step's raw-string
  desync) are all open on this gate as shipped.
- **Issue #538** — verified via `gh issue view 538 --json title` (quoted under P8 above, title:
  "fine-tune: graph arm through the materialised TrainingSet table (excised from #500 U2a)"): the graph
  fine-tune arm's own `TrainingSet` materialisation (a `RecordBatch`-sourced producer input) does not
  exist; the graph arm samples in memory, as main's does. This issue's own body is also the only source
  this contract has for the SQL-only producer construct P8 names as the excision's reason (see P8).
- **Issues #556 and #557 do not apply to this contract.** Both were checked with `gh issue view` this
  session:

```
$ gh issue view 556 --json title
{"title":"Mechanical plan-citation gate: plan docs cite symbols (path::item), resolved in CI"}
$ gh issue view 557 --json title
{"title":"Lead gate R12: CI-derived required call-site set for the anticipation record's mutations rows"}
```

  #556 is a mechanical plan-citation gate for `docs/plans/**` symbol references (the exact discipline
  this contract itself follows, by instruction); #557 is a CI-derived required-call-site reader for the
  lead's own anticipation-record `mutations` rows. Neither issue number, nor its subject matter, appears
  anywhere in either unit's diff:

```
$ grep -rn '#556\|#557\|issues/556\|issues/557' ci/ docs/plans/67-distributed-training/
(no output -- no match)
```

  They are unrelated proposals from the same program, named here only because this contract was asked
  to check for them, not because either unit leaves a residual they track.
- **The id-secrecy scan** (P2, U7b-A1-pull) is not part of PR-B1 at all — it is scheduled, by citation
  only, on the cluster-leg unit that has not yet been dispatched.
- U2a's own P8 excision left the graph fine-tune path with no independent test coverage beyond what main
  already carries for that path — the graph-arm-specific oracles this unit's own working record
  describes as removed (an anchor test, a recompute-refusal test, and library-level overlap/drop
  oracles) were tests for a mechanism this excision deletes, not tests of a mechanism this contract
  claims still exists.

---

## History

Both units carry their own committed sequence of closing-audit corrections on their respective
branches; this contract states only the shipped shape above and does not reproduce that sequence. It is
recorded on `feat/500-B-gang-local` (U7b-A1-pull's own commit sequence, base `9af28de0` through head
`288a34c7`) and on `feat/500-B-U2a` (U2a's own commit sequence, base `9db8d395` through head `06720d1e`)
— both inspected in full by this agent via `git log --oneline` over each range. This contract was
re-verified three times against A1-pull as the branch moved — `4a9f5be1`, then `f104d15a`, then
`288a34c7` — because each successive closing audit found a further finding on the SAME P6/P7 `uses:`
mechanism; P7's own text names all three heads and what changed between them. Every other section's
constructs are unchanged across the full `9af28de0..288a34c7` range for the files those sections cite
(confirmed by `git diff --stat`, run at each re-verification pass, showing only `check_gpu_prove_once.py`
and its own test file moving between `f104d15a` and `288a34c7`, and only comment-only journey-marker
cleanups in `runpod_gpu_gang.sh`/`check_cuda_run_artifacts.py`/`ci.yml` between `4a9f5be1` and
`f104d15a` — none of which moves a construct P1–P6 or P8–P12 cite). The scratchpad working documents
this contract was written from (`CONTRACT-U7b-v9.md`'s A1-pull sections and `CONTRACT-U2a-fix1.md`) are
session working material, not committed artifacts of this repository, and are not cited as a source of
truth for any claim above that this agent did not independently verify against the heads named in this
file's title.

## Citations verified against which head

Every construct cited above is tagged, by section, `(A1-pull @ 288a34c7)` or `(U2a @ 06720d1e)`, except
P7's own explicit references to the two superseded A1-pull states (`4a9f5be1`, `f104d15a`; named there
by commit, never presented as current). This agent opened each cited construct directly — `grep -n`,
`sed -n`, or `git show <head>:<path> | grep -n` — rather than trusting either scratchpad source
document's own line numbers, which predate at least one of the cited heads' own later commits. No
`path:line` form appears in
this document outside a fenced block quoting this agent's own executed command and its output.
