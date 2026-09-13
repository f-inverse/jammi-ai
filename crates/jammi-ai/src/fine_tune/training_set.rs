//! The rows a fine-tune trains on, as a **producer output** rather than a run's
//! scratch space.
//!
//! A training run no longer re-runs its source query into memory. It
//! materialises the projected rows once, through
//! [`ResultStore::materialize_training_set`](jammi_db::store::ResultStore::materialize_training_set), into an immutable Parquet result
//! table of kind `TrainingSet` carrying a definition hash and the standard
//! manifest attestation, and then reads that table back. Two runs over the same
//! source query, columns, task and format name the same table — whatever their
//! world size, batch or validation split, none of which enter the table's
//! identity.
//!
//! # The order contract has two halves, and this module owns the reader's
//!
//! The producer commits the rows sorted by the **full projected tuple**
//! ([`TRAINING_SET_ORDER_RULE_V1`](jammi_db::store::manifest::TRAINING_SET_ORDER_RULE_V1)). A Parquet scan gives no row-order
//! guarantee: the table is written with 64K row groups and the session plans at
//! `[engine] execution_threads` partitions, so a table larger than one row group
//! comes back interleaved unless the reader asks for the order. [`read_back_sql`]
//! is the one place that asks, and it renders the key through
//! [`training_set_order_by`] rather than hand-writing it — a reader that spelled
//! the direction or the NULL placement differently would read rows in an order
//! the table's own descriptor does not claim, and nothing would report it.
//!
//! # Anchors
//!
//! A registered source exposes no version or digest surface, so its rows are
//! anchored [`AnchorKind::UnpinnedAtInstant`](jammi_db::store::manifest::AnchorKind::UnpinnedAtInstant)
//! — the same honest anchor the embedding producer records for the same reason
//! (`pipeline/embedding.rs`). This module's one producer arm reads one such
//! source and records one anchor. The engine's reuse probe never matches an
//! unpinned anchor, so a training set
//! over a plain source is never reused across runs; the anchor still rides the
//! manifest, so staleness reports the same honest `Undecidable` it reports for
//! every unpinned input. The engine does own a resolver from a relation name to
//! a `ready` result table's content digest
//! (`Catalog::get_result_table` → `ResultStore::pin_current_version`), but no
//! fine-tune source can reach it: a source is resolved through
//! `SessionContext::catalog(source_id)`, and a result table is registered as the
//! BARE table `jammi.{name}` in the default catalog, never as a catalog of its
//! own. That path is therefore uncovered here rather than speculatively wired —
//! see `training_set::a_result_table_cannot_be_a_fine_tune_source` in the
//! integration suite for the executed probe.
//!
//! # The graph arm does not go through this module
//!
//! A graph fine-tune's sampled pairs are the output of a deterministic biased
//! walk, not of a query the engine can express as durable SQL, and re-running
//! that walk from the recorded spec is exactly `reconstruct_graph_loader`'s
//! job (`worker.rs`). They are sampled in memory and trained on directly, the
//! way `origin/main` always did it; no `TrainingSet` table is written, and
//! nothing here registers or deregisters a relation on the shared session for
//! them. Giving the graph arm a table of its own is
//! <https://github.com/f-inverse/jammi-ai/issues/538>.

use arrow::array::RecordBatch;
use jammi_db::error::Result;
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::manifest::InputAnchor;
use jammi_db::store::{training_set_order_by, TrainingSetSpec, TrainingSetTable};

use crate::model::ModelTask;
use crate::session::InferenceSession;

/// The SQL that reads a materialised training set back in its **committed
/// order** — the reader's half of the order contract, in one place.
///
/// `SELECT *` over the registered table (its schema is exactly the projection,
/// in declared order) with [`training_set_order_by`] re-applied. The relation is
/// spelled with [`TrainingSetTable::sql_relation`]: a result-table name carries
/// hyphens (a sanitized model id) and dots (a nanosecond timestamp), so the
/// unquoted form re-parses as arithmetic and as a multi-part reference — never
/// the table.
pub fn read_back_sql(table: &TrainingSetTable, columns: &[String]) -> String {
    format!(
        "SELECT * FROM {} {}",
        table.sql_relation(),
        training_set_order_by(columns)
    )
}

/// [`read_back_sql`]'s RANGE-scoped sibling (CONTRACT-U2b-fix1.md M3): the
/// same committed-order `ORDER BY` re-applied via [`training_set_order_by`],
/// with a `LIMIT`/`OFFSET` pair carved from `range` rather than reading the
/// whole table. **No `target_partitions` pin** — unlike
/// `data::open_row_range_stream`'s scoped, `ORDER BY`-free scan
/// (which relies on a single-partition sequential plan visiting row groups in
/// committed order because there is no sort to get wrong), this query asks
/// DataFusion to sort explicitly, so the ambient session's own partitioning
/// is free to plan however it likes — a `SortExec` (or a sort-preserving
/// merge over several partitions) makes the result correct regardless of how
/// many partitions read it.
///
/// The one caller today (`data::build_classification_loader_eager`)
/// needs exactly this: a bounded, but still fully-ordered, read of a range
/// that cannot be decoded one `RecordBatch` at a time (a chunk-at-a-time
/// classification decode is refused — see `decode_record_batch`'s doc — so
/// this reader stays an eager, whole-range-at-once query, unlike the
/// streaming scan).
///
/// `range.start`/`range.len()` become `OFFSET`/`LIMIT` by plain
/// interpolation, never a bound parameter (this crate's own convention —
/// `LIMIT` is not bindable in DataFusion's SQL front end).
pub fn read_back_range_sql(
    table: &TrainingSetTable,
    columns: &[String],
    range: std::ops::Range<usize>,
) -> String {
    format!(
        "SELECT * FROM {} {} LIMIT {} OFFSET {}",
        table.sql_relation(),
        training_set_order_by(columns),
        range.len(),
        range.start,
    )
}

/// Materialise `columns` of a registered `source` as a training set, then read
/// the committed rows back in order.
///
/// The projection runs unordered — the producer owns the sort, and asking the
/// source for an order it is about to re-impose would only plan the sort twice.
pub async fn materialize_projection(
    session: &InferenceSession,
    source_id: &str,
    columns: &[String],
    task: ModelTask,
    format: &str,
) -> Result<(TrainingSetTable, Vec<RecordBatch>)> {
    let table_name = session.find_table_name(source_id)?;
    let projection = columns
        .iter()
        .map(|c| quote_ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    let source_sql = format!(
        "SELECT {projection} FROM {}",
        source_relation(source_id, &table_name)
    );
    materialize_and_read(
        session,
        TrainingSetSpec {
            source_id,
            source_sql: &source_sql,
            columns,
            task,
            format,
            // The source has no version surface to pin, so it is anchored at
            // the instant it was read — the same honest anchor the embedding
            // producer records for the same reason, constructed here rather
            // than behind a helper so the anchor value never travels apart
            // from the read it describes.
            inputs: vec![InputAnchor::unpinned_at_instant(
                source_id,
                chrono::Utc::now().to_rfc3339(),
            )],
            device: session.compute_device(),
        },
    )
    .await
}

/// Materialise a spec through the producer and read the table back in its
/// committed order.
///
/// The read runs on the SAME `SessionContext` the verb was handed: the verb
/// binds the table there on both the computed and the reused path, so the
/// read-back resolves without a second registration and a reused table is read
/// exactly like a fresh one.
async fn materialize_and_read(
    session: &InferenceSession,
    spec: TrainingSetSpec<'_>,
) -> Result<(TrainingSetTable, Vec<RecordBatch>)> {
    let columns = spec.columns.to_vec();
    let table = session
        .result_store()
        .materialize_training_set(session.context(), spec)
        .await?;
    let batches = session.sql(&read_back_sql(&table, &columns)).await?;
    Ok((table, batches))
}
/// The reader-class allow-list (CONTRACT-U2b-fix1.md M3): every production
/// (non-test) call site of [`TrainingSetTable::sql_relation`] in this crate,
/// keyed by `path:function` rather than `path:line` — a line number drifts
/// under an unrelated edit, a function name does not — with the ONE property
/// each entry must hold: it either applies [`training_set_order_by`] itself,
/// or pins `target_partitions = 1` on a scan with no `ORDER BY` at all (the
/// only way an un-ordered read is still correct — see
/// [`super::data::open_row_range_stream`]'s doc). A caller that reads a
/// relation by name without doing one of the two loses the committed order
/// silently on a multi-row-group table scanned by more than one partition —
/// exactly the class M3 fixed (the base tree's classification fallback read
/// the relation with no order applied at all).
///
/// | `path:function`                                  | mechanism                          | behavioural order assertion |
/// |---------------------------------------------------|-------------------------------------|------------------------------|
/// | `fine_tune/training_set.rs:read_back_sql`          | `ORDER BY` via `training_set_order_by` | `training_set::read_back_re_applies_the_committed_order_across_row_groups` (`tests/it/training_set.rs`) |
/// | `fine_tune/training_set.rs:read_back_range_sql`    | `ORDER BY` via `training_set_order_by`, `LIMIT`/`OFFSET` from the range | `streaming_loader::classification_eager_fallback_preserves_committed_order_across_row_groups` (`tests/it/streaming_loader.rs`) |
/// | `fine_tune/data.rs:open_row_range_stream`          | NO `ORDER BY`; `target_partitions = 1` pinned | `streaming_loader::streamed_order_matches_committed_order_at_target_partitions_one_and_n` + `streaming_loader::removing_the_target_partitions_pin_breaks_order_on_a_multi_row_group_table` (`tests/it/streaming_loader.rs`) |
///
/// This test finds every call site itself (never hand-transcribes the count)
/// by walking every `crates/*/src/**/*.rs` file from the workspace root and
/// grepping for the reader method's invocation syntax on a receiver — so a
/// NEW caller anywhere in the workspace, not just this crate, fails it, and a
/// call site that moves to a different function name (rename) requires a
/// conscious edit to this allow-list rather than silently staying "covered".
/// The needle is assembled at runtime (never spelled as one contiguous
/// literal in this module's own source) so this scan does not match its own
/// doc comments, messages, or the `const` below.
#[cfg(test)]
mod reader_class_allow_list {
    /// `(workspace-relative path, enclosing function name)` for every
    /// production call site this fold has audited and accepted.
    const ALLOWED: &[(&str, &str)] = &[
        (
            "crates/jammi-ai/src/fine_tune/training_set.rs",
            "read_back_sql",
        ),
        (
            "crates/jammi-ai/src/fine_tune/training_set.rs",
            "read_back_range_sql",
        ),
        (
            "crates/jammi-ai/src/fine_tune/data.rs",
            "open_row_range_stream",
        ),
    ];

    /// The invocation this scan looks for, assembled from two literal parts
    /// so the exact contiguous text never appears once in this file (which
    /// would otherwise match itself, its own doc comments, and its own
    /// messages).
    fn needle() -> String {
        format!(".{}{}", "sql_relation", "()")
    }

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("crates/jammi-ai/../.. must be the workspace root")
    }

    /// Every `crates/*/src/**/*.rs` file under the workspace root — `src/`
    /// only, so a test fixture calling the reader method (there are several,
    /// deliberately, to build committed-order oracles) never enters this
    /// production-code sweep.
    fn all_workspace_src_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let crates_dir = root.join("crates");
        for crate_entry in std::fs::read_dir(&crates_dir)
            .unwrap_or_else(|e| panic!("read_dir({}): {e}", crates_dir.display()))
        {
            let crate_entry = crate_entry.unwrap();
            if !crate_entry.file_type().unwrap().is_dir() {
                continue;
            }
            let src_dir = crate_entry.path().join("src");
            if src_dir.is_dir() {
                walk_rs_files(&src_dir, &mut out);
            }
        }
        out
    }

    fn walk_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()))
        {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk_rs_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// The name of the nearest `fn`/`async fn` declaration at or before
    /// `line_idx` (0-based) in `lines` — a plain textual scan, adequate for
    /// this codebase's style of one function body per reader-method call
    /// site (never a closure or a nested `fn`).
    fn enclosing_fn_name(lines: &[&str], line_idx: usize) -> Option<String> {
        let fn_line = regex_lite_find_fn(lines, line_idx)?;
        let after_fn = fn_line.split("fn ").nth(1)?;
        let name: String = after_fn
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    /// Walk backward from `line_idx` for a line containing `"fn "` — no
    /// external regex dependency needed for this narrow a scan.
    fn regex_lite_find_fn<'a>(lines: &[&'a str], line_idx: usize) -> Option<&'a str> {
        (0..=line_idx).rev().map(|i| lines[i]).find(|l| {
            l.trim_start().starts_with("fn ")
                || l.trim_start().starts_with("pub fn ")
                || l.trim_start().starts_with("pub(crate) fn ")
                || l.trim_start().starts_with("async fn ")
                || l.trim_start().starts_with("pub async fn ")
                || l.trim_start().starts_with("pub(crate) async fn ")
        })
    }

    #[test]
    fn every_production_sql_relation_call_site_is_on_the_allow_list() {
        let root = workspace_root();
        let needle = needle();
        let mut found: Vec<(String, String)> = Vec::new();
        for path in all_workspace_src_files(&root) {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                // Skip comment/doc lines outright — a mention of the reader
                // method in prose is not an invocation of it.
                if line.trim_start().starts_with("//") {
                    continue;
                }
                if line.contains(&needle) {
                    let rel = path
                        .strip_prefix(&root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    let func = enclosing_fn_name(&lines, i).unwrap_or_else(|| {
                        panic!(
                            "{rel}:{}: reader-method call with no enclosing `fn` found by this \
                             scan — widen `regex_lite_find_fn`'s prefix list",
                            i + 1
                        )
                    });
                    found.push((rel, func));
                }
            }
        }
        found.sort();
        found.dedup();
        let mut allowed: Vec<(String, String)> = ALLOWED
            .iter()
            .map(|(p, f)| (p.to_string(), f.to_string()))
            .collect();
        allowed.sort();

        let extra: Vec<_> = found.iter().filter(|e| !allowed.contains(e)).collect();
        assert!(
            extra.is_empty(),
            "new caller(s) of `TrainingSetTable::sql_relation()` not on the reader-class \
             allow-list — each one must either apply `training_set_order_by` or pin \
             `target_partitions = 1` on an `ORDER BY`-free scan, then be added here with its \
             own behavioural order assertion: {extra:?}"
        );
        let missing: Vec<_> = allowed.iter().filter(|e| !found.contains(e)).collect();
        assert!(
            missing.is_empty(),
            "allow-listed call site(s) no longer found — the allow-list is stale, narrow it: \
             {missing:?}"
        );
    }
}

