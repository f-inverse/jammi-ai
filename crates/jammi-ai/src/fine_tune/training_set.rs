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

/// The single constructor every production call site in this crate builds a
/// [`TrainingSetSpec`] through: a future field added to the spec is added in
/// exactly ONE place, rather than re-derived independently at each of
/// [`materialize_projection`] and `pipeline/recompute.rs`'s
/// `recompute_training_set`.
///
/// A thin pass-through by design — it changes nothing about what a caller
/// supplies, only WHERE the seven fields are named — so it cannot move a
/// [`TrainingSetSpec::definition_hash`]; pinned by `training_set_spec_matches_
/// a_hand_built_spec_byte_for_byte` below.
pub(crate) fn training_set_spec<'a>(
    source_id: &'a str,
    source_sql: &'a str,
    columns: &'a [String],
    task: ModelTask,
    format: &'a str,
    inputs: Vec<InputAnchor>,
    device: jammi_db::store::manifest::ComputeDevice,
) -> TrainingSetSpec<'a> {
    TrainingSetSpec {
        source_id,
        source_sql,
        columns,
        task,
        format,
        inputs,
        device,
    }
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
        training_set_spec(
            source_id,
            &source_sql,
            columns,
            task,
            format,
            // The source has no version surface to pin, so it is anchored at
            // the instant it was read — the same honest anchor the embedding
            // producer records for the same reason.
            vec![InputAnchor::unpinned_at_instant(
                source_id,
                chrono::Utc::now().to_rfc3339(),
            )],
            session.compute_device(),
        ),
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

/// The reader-class allow-list: every production (non-test) call site that
/// reaches a training-set table's relation KEY in this crate, keyed by
/// `path:function` rather than `path:line` — a line number drifts under an
/// unrelated edit, a function name does not — with the ONE property each
/// entry must hold: it applies [`training_set_order_by`] itself. A caller
/// that reads a relation by name without doing so loses the committed order
/// silently on a multi-row-group table scanned by more than one partition.
///
/// # The class this scan covers: every ROUTE to the relation key, not one name
///
/// [`TrainingSetTable::sql_relation`] is not the only way to reach the
/// registered name — the scan matches every route:
/// - `.sql_relation(` — the dot-call form.
/// - `sql_relation(&` — the UFCS form (`TrainingSetTable::sql_relation(&t)`).
/// - `registered_name(` — [`TrainingSetTable::registered_name`], the
///   UNQUOTED key `sql_relation` itself quotes. Its own doc says it is "NOT
///   safe to interpolate into SQL as-is", but a caller that reaches for it
///   directly (skipping the quoting) is still on the identical route to the
///   same relation, and the order hazard is the same.
///
/// [`TrainingSetTable::table_name`] is deliberately NOT on this scan: it
/// returns the bare catalog name with no `jammi.` schema prefix, so it
/// cannot stand in for either route above without a caller re-deriving the
/// missing prefix and quoting by hand — no such caller exists in this crate
/// today, and one that started would be re-implementing
/// `sql_relation`/`registered_name`, which brings it onto this scan the
/// moment it calls either.
///
/// One exclusion, by construction rather than by allow-listing: `sql_relation`'s
/// OWN body (`crates/jammi-db/src/store/mod.rs`) calls `registered_name()` on
/// itself to build the string it then quotes — that call constructs an
/// identifier, not a query, so there is no order to lose. The scan skips
/// matches whose enclosing function IS `table_name`/`registered_name`/
/// `sql_relation` in that one file (the accessors' own implementations),
/// never a caller elsewhere.
///
/// | `path:function`                                  | mechanism                          | behavioural order assertion |
/// |---------------------------------------------------|-------------------------------------|------------------------------|
/// | `fine_tune/training_set.rs:read_back_sql`          | `ORDER BY` via `training_set_order_by` | `training_set::read_back_re_applies_the_committed_order_across_row_groups` (`tests/it/training_set.rs`) |
///
/// This test finds every call site itself (never hand-transcribes the count)
/// by walking every `crates/*/src/**/*.rs` file from the workspace root and
/// grepping for each route's invocation syntax — so a NEW caller anywhere in
/// the workspace, not just this crate, fails it, and a call site that moves
/// to a different function name (rename) requires a conscious edit to this
/// allow-list rather than silently staying "covered". Every needle is
/// assembled at runtime (never spelled as one contiguous literal in this
/// module's own source) so this scan does not match its own doc comments,
/// messages, or the `const` below.
#[cfg(test)]
mod reader_class_allow_list {
    /// `(workspace-relative path, enclosing function name)` for every
    /// production call site this fold has audited and accepted.
    const ALLOWED: &[(&str, &str)] = &[(
        "crates/jammi-ai/src/fine_tune/training_set.rs",
        "read_back_sql",
    )];

    /// The accessors' own implementations (`crates/jammi-db/src/store/mod.rs`)
    /// — excluded by construction, not by allow-listing, since a match there
    /// is the method building its own return value, never a caller reaching
    /// for the relation key. See the module doc's "One exclusion" note.
    const ACCESSOR_IMPL_FILE: &str = "crates/jammi-db/src/store/mod.rs";
    const ACCESSOR_IMPL_FNS: &[&str] = &["table_name", "registered_name", "sql_relation"];

    /// Every route this scan matches, each assembled from separate literal
    /// parts so the exact contiguous text never appears once in this file
    /// (which would otherwise match itself, its own doc comments, and its
    /// own messages).
    fn needles() -> Vec<String> {
        vec![
            format!(".{}(", "sql_relation"),
            format!("{}(&", "sql_relation"),
            format!("{}(", "registered_name"),
        ]
    }

    /// A route's own definition line (`fn sql_relation(` / `fn registered_name(`)
    /// is not a call site — the return-type accessor being DEFINED, never
    /// invoked. Distinct from [`ACCESSOR_IMPL_FNS`]'s exclusion, which covers
    /// calls made FROM inside those functions' bodies.
    fn is_definition_line(line: &str) -> bool {
        line.contains(&format!("fn {}(", "sql_relation"))
            || line.contains(&format!("fn {}(", "registered_name"))
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
        let needles = needles();
        let mut found: Vec<(String, String)> = Vec::new();
        for path in all_workspace_src_files(&root) {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let lines: Vec<&str> = text.lines().collect();
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            for (i, line) in lines.iter().enumerate() {
                // Skip comment/doc lines outright — a mention of a route in
                // prose is not an invocation of it.
                if line.trim_start().starts_with("//") {
                    continue;
                }
                // A route's own definition line is not a call site.
                if is_definition_line(line) {
                    continue;
                }
                if !needles.iter().any(|n| line.contains(n)) {
                    continue;
                }
                let func = enclosing_fn_name(&lines, i).unwrap_or_else(|| {
                    panic!(
                        "{rel}:{}: reader-method call with no enclosing `fn` found by this \
                         scan — widen `regex_lite_find_fn`'s prefix list",
                        i + 1
                    )
                });
                // The accessors' own bodies (`sql_relation` calling
                // `registered_name` on itself) are excluded by construction —
                // see the module doc's "One exclusion" note.
                if rel == ACCESSOR_IMPL_FILE && ACCESSOR_IMPL_FNS.contains(&func.as_str()) {
                    continue;
                }
                found.push((rel.clone(), func));
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
            "new caller(s) reaching a training-set table's relation key (via `sql_relation`, \
             its UFCS form, or `registered_name`) not on the reader-class allow-list — each \
             one must either apply `training_set_order_by` or pin `target_partitions = 1` on \
             an `ORDER BY`-free scan, then be added here with its own behavioural order \
             assertion: {extra:?}"
        );
        let missing: Vec<_> = allowed.iter().filter(|e| !found.contains(e)).collect();
        assert!(
            missing.is_empty(),
            "allow-listed call site(s) no longer found — the allow-list is stale, narrow it: \
             {missing:?}"
        );
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// CONTRACT-U2b-fix1.md item 3: [`training_set_spec`] is a thin
    /// pass-through, so it must name the exact same [`TrainingSetSpec::
    /// definition_hash`] as a hand-built struct literal over the identical
    /// seven fields — the "unification must not change any hash" property,
    /// pinned directly rather than by re-running a whole fixture through the
    /// engine.
    #[test]
    fn training_set_spec_matches_a_hand_built_spec_byte_for_byte() {
        let columns = vec!["anchor".to_string(), "positive".to_string()];
        let inputs = vec![InputAnchor::unpinned_at_instant(
            "training",
            "2024-01-01T00:00:00Z".to_string(),
        )];
        let device = jammi_db::store::manifest::ComputeDevice::Cpu;

        let via_helper = training_set_spec(
            "training",
            "SELECT anchor, positive FROM training",
            &columns,
            ModelTask::TextEmbedding,
            "pairs",
            inputs.clone(),
            device.clone(),
        );
        let hand_built = TrainingSetSpec {
            source_id: "training",
            source_sql: "SELECT anchor, positive FROM training",
            columns: &columns,
            task: ModelTask::TextEmbedding,
            format: "pairs",
            inputs,
            device,
        };
        assert_eq!(
            via_helper.definition_hash().unwrap(),
            hand_built.definition_hash().unwrap(),
            "training_set_spec must be a pure pass-through: it cannot move the definition hash \
             relative to constructing the SAME fields directly"
        );
    }

}
