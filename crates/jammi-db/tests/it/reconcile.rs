//! `ResultStore::reconcile` / `reconcile_all` — the object-store cross-check
//! against the catalog. Every oracle here is engine-level (`file://` and
//! `memory://`, SQLite catalog); the wire/CLI surface lands in a later
//! commit.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ComputeDevice, ComputePrecision, MaterializationEnv, ModelContentDigest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{
    BuildingTable, EmbeddingTableSpec, Materialization, ReconcileOptions, ResultStore,
};
use jammi_db::TenantId;
use tempfile::tempdir;
use uuid::Uuid;

const DIMS: usize = 4;

fn fresh_tenant() -> TenantId {
    TenantId::from_uuid(Uuid::new_v4()).unwrap()
}

fn descriptor() -> ProducingDescriptor {
    ProducingDescriptor::Embedding {
        model_id: "test-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: "docs".into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: DIMS,
    }
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "test-model".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("reconcile-fixture-digest".into()),
            quantization: None,
        }],
    )
}

fn sample_rows(n: usize) -> Vec<(String, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let vec = (0..DIMS).map(|d| (i * DIMS + d) as f32 + 1.0).collect();
            (format!("row-{i}"), vec)
        })
        .collect()
}

async fn materialize_healthy_table(
    store: &ResultStore,
    ctx: &SessionContext,
    source_id: &str,
) -> jammi_db::catalog::result_repo::ResultTableRecord {
    let rows = sample_rows(5);
    store
        .materialize_embedding_table(
            ctx,
            EmbeddingTableSpec {
                source_id,
                model_id: "test-model",
                derived_from: None,
                dimensions: DIMS,
                key_column: Some("_row_id"),
                text_columns: Some("body"),
            },
            &rows,
            Materialization::new(&descriptor(), &env(), vec![]),
        )
        .await
        .unwrap()
}

/// Backdate the mtime of every regular file directly under `dir` by
/// `by` — lets a test manufacture an "already past grace" orphan candidate
/// deterministically, with no real-time sleep.
fn backdate_dir(dir: &std::path::Path, by: Duration) {
    let target = std::time::SystemTime::now() - by;
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::File::options()
                .write(true)
                .open(entry.path())
                .unwrap()
                .set_modified(target)
                .unwrap();
        }
    }
}

/// The lease-duration floor `apply=true` must respect. This test never
/// actually waits for expiry — it only checks the synchronous config
/// guard — so a short-but-valid whole-second pair
/// (`LeaseConfig::intervals`'s only public constructor) is enough.
fn short_lease() -> jammi_db::catalog::lease::LeaseIntervals {
    jammi_db::config::LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    }
    .intervals()
    .unwrap()
}

/// Register a `building` embedding row and write a valid, closed Parquet
/// under it directly — bypassing the catalog's `building -> ready` flip, so
/// the row stays `building` with real bytes on disk and NO
/// `.materialization.json` sidecar: the "torn write before manifest" shape
/// [`ResultStore::recover`]'s expired-building pre-pass reaps via
/// `reap_after_fail_cas` (fails the row, then deletes the Parquet).
async fn create_building_embedding_with_parquet(
    store: &ResultStore,
    source_id: &str,
    n: usize,
) -> BuildingTable {
    create_building_embedding_with_parquet_and_catalog_dims(store, source_id, n, Some(DIMS as i32))
        .await
}

/// Same construction as [`create_building_embedding_with_parquet`], but the
/// catalog row's own `dimensions` column is `catalog_dims` — independent of
/// the Parquet's PHYSICAL vector width, which is always [`DIMS`] here. Lets a
/// test manufacture the degenerate `dimensions == 0` / `NULL` catalog state
/// (esc-484) without needing a real zero-width vector column, which nothing
/// downstream of `classify_expired_row`'s `dimensions == 0` early return ever
/// reads.
async fn create_building_embedding_with_parquet_and_catalog_dims(
    store: &ResultStore,
    source_id: &str,
    n: usize,
    catalog_dims: Option<i32>,
) -> BuildingTable {
    let info = store
        .create_table(
            source_id,
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            catalog_dims,
            Some("_row_id"),
            None,
        )
        .await
        .unwrap();

    let schema = embedding_table_schema(DIMS);
    let row_ids: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    let row_id_arr = StringArray::from_iter_values(row_ids.iter().map(|s| s.as_str()));
    let source_arr = StringArray::from_iter_values((0..n).map(|_| source_id));
    let model_arr = StringArray::from_iter_values((0..n).map(|_| "test-model"));
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| (0..DIMS).map(move |d| (i * DIMS + d) as f32))
        .collect();
    let item = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        false,
    ));
    let vectors =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(row_id_arr),
            Arc::new(source_arr),
            Arc::new(model_arr),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(info.parquet_url(), schema).await.unwrap();
    if n > 0 {
        writer.write_batch(&batch).await.unwrap();
    }
    writer.close().await.unwrap();
    info
}

// ─── report-count correctness: the expired-building pre-pass reports
//     EVERY byte it reclaims (or, under a dry-run, would reclaim) EXACTLY
//     ONCE — never zero, never twice. Before this fix (a2bea619's "double-
//     counted" rationale, which moved the pre-pass BEFORE the listing so a
//     key it deleted could never appear in `listed` at all) `apply=true`
//     reaped the row's Parquet but reported it in NO field: `orphans` was
//     empty and `bytes_reclaimed` was `0` for a pass that had just reclaimed
//     real bytes. RED against aa117dbf (pasted below) proves the old
//     oracle pinned exactly that silence, not a real absence of double
//     counting. ─────────────────────────────────────────────────────────────

/// Build a `building` row whose lease has expired with a valid Parquet but
/// no manifest sidecar (a torn write before the `building -> ready` flip) —
/// the exact state [`ResultStore::reconcile`]'s expired-building pre-pass
/// reaps via `reap_after_fail_cas` (fails the row, deletes the Parquet).
/// Backdated so the SAME object would also qualify as a past-grace orphan
/// candidate through the ordinary object→row arm if it were ever (wrongly)
/// re-listed there — the condition that would double-count it absent the
/// `reaped` set this fix introduces. Returns the table name and the
/// Parquet's local filesystem path.
async fn torn_building_row_fixture(
    store: &ResultStore,
    catalog: &Catalog,
    source_id: &str,
) -> (String, String) {
    let info = create_building_embedding_with_parquet(store, source_id, 5).await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    // `abandon_building` asserts the row is `building` under a live lease,
    // detaches the writer's handle (so no background heartbeat can renew it
    // out from under the next line), THEN forces the lease into the past.
    let table_name = jammi_test_utils::abandon_building(catalog, info).await;
    let table_dir = std::path::Path::new(&parquet_local).parent().unwrap();
    backdate_dir(table_dir, Duration::from_secs(3600));
    (table_name, parquet_local)
}

/// Same construction as [`torn_building_row_fixture`] — an expired-lease
/// `building` row with a valid Parquet and no manifest sidecar — but WITHOUT
/// backdating: the bytes are as fresh as `Utc::now()`, well inside any
/// `grace` this suite uses. Isolates the pre-pass's CAS-licensed reap (never
/// age-gated) from the general object→row arm's `grace` age-gate: a torn row
/// this young must still be an orphan candidate, never `pending`, because the
/// pre-pass consults the row's expired LEASE, never the object's age.
async fn torn_building_row_fixture_young(
    store: &ResultStore,
    catalog: &Catalog,
    source_id: &str,
) -> (String, String) {
    let info = create_building_embedding_with_parquet(store, source_id, 5).await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_name = jammi_test_utils::abandon_building(catalog, info).await;
    (table_name, parquet_local)
}

/// Build an expired-lease `building` row with a valid, closed Parquet AND its
/// `.materialization.json` manifest sidecar already on disk — the exact
/// PROMOTE state [`ResultStore::recover`]'s expired-building arm claims and
/// promotes to `ready` (never reaps). Backdated past grace so the ONLY reason
/// its objects are absent from `orphans` is the pre-pass's `Promote`
/// classification protecting them, never merely a fresh-object age-gate pass
/// (which a non-backdated fixture would satisfy trivially, masking the very
/// bug this state exists to catch). Returns the table name and the Parquet's
/// local filesystem path.
async fn promotable_building_row_fixture(
    store: &ResultStore,
    catalog: &Catalog,
    source_id: &str,
) -> (String, String) {
    let info = create_building_embedding_with_parquet(store, source_id, 5).await;
    jammi_test_utils::write_manifest_sidecar_for(store, info.parquet_url(), source_id, DIMS).await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_name = jammi_test_utils::abandon_building(catalog, info).await;
    let table_dir = std::path::Path::new(&parquet_local).parent().unwrap();
    backdate_dir(table_dir, Duration::from_secs(3600));
    (table_name, parquet_local)
}

/// A built one-segment index over `n` synthetic rows, at the store's default
/// precision — the shape [`BuildingTable::append_segment`] persists. Rows are
/// independent of [`sample_rows`]'s own ids/vectors: a stale-segment fixture
/// only needs real bytes on disk, never rows that match the table's current
/// Parquet.
fn built_segment_index(n: usize) -> SidecarIndex {
    let mut idx = SidecarIndex::new(
        DIMS,
        &AnnIndexConfig::default(),
        AnnIndexConfig::default().storage_precision,
    )
    .unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..DIMS).map(|d| (i * DIMS + d) as f32).collect();
        idx.add(&format!("seg-row-{i}"), &v).unwrap();
    }
    idx.build().unwrap();
    idx
}

/// Every sidecar file segment `seg` of `table_name` actually wrote under
/// `table_dir`, paired with its true on-disk size — read directly off the
/// filesystem (never assumed from
/// [`jammi_db::storage::sidecar_layout::sidecar_extensions`]'s full,
/// precision-gated extension list) so a test's expected-key set can never
/// drift from what really landed.
fn segment_sidecar_files(
    table_dir: &std::path::Path,
    table_name: &str,
    seg: i64,
) -> Vec<(String, u64)> {
    let prefix = format!("{table_name}__seg{seg}.");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(table_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(&prefix) {
            out.push((name, entry.metadata().unwrap().len()));
        }
    }
    assert!(
        !out.is_empty(),
        "fixture sanity: segment {seg} of '{table_name}' must have written real sidecar files \
         under {table_dir:?}"
    );
    out
}

// ─── #484 design revision: a promotion is not a reclaim. The promote arm's
//     rebuild still purges the row's CURRENT segment set and rewrites, at
//     most, a single fresh segment 0 — but a stale sibling the rebuild
//     deletes and does not rewrite is now reported NOWHERE this pass, in
//     EITHER mode: it is the promotion's own bookkeeping, never a reclaim
//     this pass credits or previews. A LATER pass, once such bytes truly
//     survive on disk unreferenced and past grace (the rebuild-fails-after-
//     the-purge case, pinned separately below), reclaims them normally. ────

/// Two segments (ids 0 and 1) already appended before the crash: the
/// rebuild purges both and rewrites exactly one fresh segment 0. Segment 1's
/// sidecars are real deletions but NOT a reclaim this pass reports — dry-run
/// protects the row's WHOLE existing key set wholesale (predicting nothing
/// about what the rebuild will purge); apply excludes what its own rebuild
/// actually purged from every accounting field. Both report NOTHING for this
/// row's segments in this pass.
#[tokio::test]
async fn promote_of_a_stale_second_segment_reports_nothing_this_pass() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let info = create_building_embedding_with_parquet(&store, "docs-two-seg", 5).await;
    info.append_segment(&built_segment_index(2)).await.unwrap();
    info.append_segment(&built_segment_index(3)).await.unwrap();
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "docs-two-seg", DIMS)
        .await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_dir = std::path::Path::new(&parquet_local)
        .parent()
        .unwrap()
        .to_path_buf();
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;

    // Read the real sidecar files off disk BEFORE either pass runs — these
    // are the exact keys this test asserts on.
    let seg1_files = segment_sidecar_files(&table_dir, &table_name, 1);
    let seg0_files = segment_sidecar_files(&table_dir, &table_name, 0);

    backdate_dir(&table_dir, Duration::from_secs(3600));

    let dry = store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    for (name, _) in seg1_files.iter().chain(seg0_files.iter()) {
        assert!(
            !dry.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "a promotion is not a reclaim: a dry-run must report NOTHING for a promoted row's \
             segment sidecars, stale or surviving: {dry:?}"
        );
        assert!(
            !dry.pending.iter().any(|o| o.ends_with(name.as_str())),
            "{dry:?}"
        );
    }
    assert_eq!(
        dry.bytes_reclaimed, 0,
        "a dry-run must never predict a promotion's own rebuild as a reclaim: {dry:?}"
    );
    for (name, _) in seg1_files.iter().chain(seg0_files.iter()) {
        assert!(
            table_dir.join(name).exists(),
            "a dry-run must never delete anything: '{name}' missing"
        );
    }

    let apply = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    let promoted = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        promoted.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "the row must be promoted, not reaped: {apply:?}"
    );
    for (name, _) in &seg1_files {
        assert!(
            !table_dir.join(name).exists(),
            "apply must actually delete segment 1's stale sidecar '{name}' — the promotion's \
             rebuild still purges it for real, even though this pass reports nothing for it"
        );
    }
    for (name, _) in &seg0_files {
        assert!(
            table_dir.join(name).exists(),
            "segment 0's sidecar '{name}' must survive apply (rewritten at the same key)"
        );
    }
    let segs = store
        .catalog()
        .list_index_segments(&table_name)
        .await
        .unwrap();
    assert_eq!(segs.len(), 1, "exactly one rebuilt segment: {segs:?}");

    for (name, _) in seg1_files.iter().chain(seg0_files.iter()) {
        assert!(
            !apply.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "a promotion is not a reclaim: apply must report NOTHING for this row's segment \
             sidecars, in this pass, even though segment 1's bytes are truly gone: {apply:?}"
        );
    }
    assert_eq!(
        apply.bytes_reclaimed, 0,
        "apply must never credit a promotion's own rebuild as a reclaim: {apply:?}"
    );

    assert_eq!(dry.orphans, apply.orphans, "{dry:?} vs {apply:?}");
    assert_eq!(dry.orphan_count, apply.orphan_count, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.bytes_reclaimed, apply.bytes_reclaimed,
        "{dry:?} vs {apply:?}"
    );
}

/// #484 design revision, second half: when a promotion's rebuild fails
/// AFTER `purge_segments` has run, the purged keys are excluded from THIS
/// pass's accounting the same way a successful promotion's are — but a key
/// `purge_segments` itself FAILS to delete (manufactured here with an
/// unwritable table directory, exactly like `abort_aggregates_a_real_delete_
/// failure_into_one_error`) survives on disk and is picked up normally by
/// the ordinary age-gated arm once it is genuinely unreferenced: `pending`
/// while the directory is still unwritable (a real delete failure, never
/// silently swallowed), then `orphans` — with its TRUE size — once a LATER
/// pass runs with the directory writable again.
///
/// `append_segment`'s own catalog insert (unconditional) precedes its
/// physical sidecar write, so `write_fresh_segment_zero`'s failed write
/// under the SAME permission denial leaves a dangling `index_segments` row
/// for a fresh segment `0` — at the SAME `index_path` the ORIGINAL segment
/// `0` used, so the ORIGINAL (unwritten-over, since the write also failed)
/// segment `0` bytes stay legitimately referenced/protected forever after,
/// an existing `append_segment` non-atomicity this fix neither causes nor
/// remedies. Segment `1`'s catalog row has no such replacement — it is
/// genuinely unreferenced once its row is purged — so it, alone, is what a
/// later pass reclaims.
#[tokio::test]
async fn rebuild_failure_after_the_purge_defers_seg1_reclaim_to_a_later_pass() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let info = create_building_embedding_with_parquet(&store, "docs-rebuild-fail", 5).await;
    info.append_segment(&built_segment_index(2)).await.unwrap();
    info.append_segment(&built_segment_index(3)).await.unwrap();
    jammi_test_utils::write_manifest_sidecar_for(
        &store,
        info.parquet_url(),
        "docs-rebuild-fail",
        DIMS,
    )
    .await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_dir = std::path::Path::new(&parquet_local)
        .parent()
        .unwrap()
        .to_path_buf();
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;

    let seg0_files = segment_sidecar_files(&table_dir, &table_name, 0);
    let seg1_files = segment_sidecar_files(&table_dir, &table_name, 1);

    backdate_dir(&table_dir, Duration::from_secs(3600));

    // Deleting (and writing) a file requires WRITE permission on its
    // containing directory (POSIX semantics) — strip it so `purge_segments`'
    // own deletes AND `write_fresh_segment_zero`'s write both fail with a
    // real I/O error, never a mere 404: the rebuild fails after the purge
    // ATTEMPTED (and failed) to clear the stale segment set.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&table_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

    let first = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    // Restore write permission unconditionally before any assertion can
    // panic, so the tempdir's own Drop can clean up either way.
    std::fs::set_permissions(&table_dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    let promoted = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        promoted.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "the row still promotes even though its rebuild failed: {first:?}"
    );
    // The bytes physically survive — the permission denial blocked the
    // unlink, not merely the report.
    for (name, _) in seg0_files.iter().chain(seg1_files.iter()) {
        assert!(
            table_dir.join(name).exists(),
            "a real delete failure must leave '{name}' in place for a later pass to retry"
        );
    }
    // This pass never credits them as reclaimed (a real delete failure is
    // never silently swallowed into a false credit).
    for (name, _) in seg0_files.iter().chain(seg1_files.iter()) {
        assert!(
            !first.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "a failed delete must never be credited as reclaimed: {first:?}"
        );
    }

    // A LATER pass, with the directory writable again, reclaims segment 1's
    // surviving bytes normally — genuinely unreferenced (its catalog row was
    // purged with no replacement) and already well past grace.
    let second = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    for (name, _) in &seg1_files {
        assert!(
            second.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "a later pass must reclaim the surviving sidecar '{name}' as an ordinary orphan: \
             {second:?}"
        );
        assert!(
            !table_dir.join(name).exists(),
            "the later pass must actually delete '{name}'"
        );
    }
    let expected_bytes: u64 = seg1_files.iter().map(|(_, size)| size).sum();
    assert_eq!(
        second.bytes_reclaimed, expected_bytes,
        "the later pass must report segment 1's TRUE combined size: {second:?}"
    );
    // Segment 0's ORIGINAL bytes remain protected — `append_segment`'s
    // dangling re-insert at the SAME `index_path` (its physical write itself
    // failed under the same permission denial) means the catalog still
    // names this key as a current segment.
    for (name, _) in &seg0_files {
        assert!(
            table_dir.join(name).exists(),
            "segment 0's original bytes are untouched by this whole reproducer: '{name}' missing"
        );
    }
}

/// One segment already appended, but the row's CURRENT Parquet carries ZERO
/// rows: the rebuild purges the stale segment and, because
/// `index.len() == 0`, rewrites NOTHING. Segment 0's sidecars are real
/// deletions but NOT a reclaim this pass reports — same rule as the
/// two-segment sibling above. A later, SEPARATE reconcile pass (this time,
/// past a real delete failure — an unwritable table directory, exactly like
/// `rebuild_failure_after_the_purge_defers_seg1_reclaim_to_a_later_pass`)
/// still reclaims the surviving bytes as an ordinary orphan.
#[tokio::test]
async fn promote_over_a_zero_row_parquet_reports_nothing_this_pass() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // Zero rows in the Parquet itself; a stale segment appended anyway (the
    // shape a truncated-then-rewritten-empty crash could leave).
    let info = create_building_embedding_with_parquet(&store, "docs-zero-row", 0).await;
    info.append_segment(&built_segment_index(2)).await.unwrap();
    jammi_test_utils::write_manifest_sidecar_for(&store, info.parquet_url(), "docs-zero-row", DIMS)
        .await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_dir = std::path::Path::new(&parquet_local)
        .parent()
        .unwrap()
        .to_path_buf();
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;

    let seg0_files = segment_sidecar_files(&table_dir, &table_name, 0);

    backdate_dir(&table_dir, Duration::from_secs(3600));

    let dry = store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    for (name, _) in &seg0_files {
        assert!(
            !dry.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "a promotion is not a reclaim: a dry-run must report NOTHING for the zero-row row's \
             stale segment 0 sidecar '{name}': {dry:?}"
        );
        assert!(
            !dry.pending.iter().any(|o| o.ends_with(name.as_str())),
            "{dry:?}"
        );
    }
    assert_eq!(dry.bytes_reclaimed, 0, "{dry:?}");
    for (name, _) in &seg0_files {
        assert!(
            table_dir.join(name).exists(),
            "a dry-run must never delete anything: '{name}' missing"
        );
    }

    let apply = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    let promoted = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        promoted.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "a zero-row Parquet with its manifest present is still promoted, never reaped: {apply:?}"
    );
    assert_eq!(promoted.row_count, 0, "{apply:?}");
    for (name, _) in &seg0_files {
        assert!(
            !table_dir.join(name).exists(),
            "apply must actually delete the stale segment 0 sidecar '{name}': it is never \
             rewritten over a zero-row Parquet"
        );
        assert!(
            !apply.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "a promotion is not a reclaim: apply must report NOTHING for this row's segment 0 \
             sidecars, in this pass, even though the bytes are truly gone: {apply:?}"
        );
    }
    let segs = store
        .catalog()
        .list_index_segments(&table_name)
        .await
        .unwrap();
    assert!(
        segs.is_empty(),
        "a zero-row Parquet's rebuild writes no segment at all: {segs:?}"
    );

    assert_eq!(dry.orphans, apply.orphans, "{dry:?} vs {apply:?}");
    assert_eq!(dry.orphan_count, apply.orphan_count, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.bytes_reclaimed, apply.bytes_reclaimed,
        "{dry:?} vs {apply:?}"
    );
}

/// esc-484 follow-up: a `dimensions == 0` catalog row is the SAME early
/// return `rebuild_index_from_parquet` itself takes — `purge_segments` is
/// never even called, so the row's WHOLE current segment set is left
/// exactly where it is. `classify_expired_row`'s `keeps` gate must mirror
/// that early return, not just the `is_embedding` / row-count pair: a stale
/// segment's sidecars must never be previewed OR actually reclaimed for such
/// a row, in EITHER mode.
#[tokio::test]
async fn dimensions_zero_row_is_untouched_for_reclaim_in_both_modes() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // Catalog `dimensions = 0` (the degenerate/NULL-ish boundary), a valid
    // 5-row Parquet, and TWO stale segments appended anyway (the shape a
    // crash right before `dimensions` was ever meaningfully set could
    // leave). Two segments — not one — so a `keeps` gate that mirrors only
    // `is_embedding && row_count > 0` (dropping the `dimensions == 0` early
    // return) is distinguishable from the correct gate: such a gate would
    // still add segment 0 to `keeps` (mistaking this row for the ordinary
    // "rewrite segment 0" case) while leaving segment 1 as a predicted
    // reclaim — a real disagreement with the actual rebuild, which never
    // purges ANYTHING for a `dimensions == 0` row.
    let info = create_building_embedding_with_parquet_and_catalog_dims(
        &store,
        "docs-dims-zero",
        5,
        Some(0),
    )
    .await;
    info.append_segment(&built_segment_index(2)).await.unwrap();
    info.append_segment(&built_segment_index(3)).await.unwrap();
    jammi_test_utils::write_manifest_sidecar_for(
        &store,
        info.parquet_url(),
        "docs-dims-zero",
        DIMS,
    )
    .await;
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_dir = std::path::Path::new(&parquet_local)
        .parent()
        .unwrap()
        .to_path_buf();
    let table_name = jammi_test_utils::abandon_building(&catalog, info).await;

    let seg0_files = segment_sidecar_files(&table_dir, &table_name, 0);
    let seg1_files = segment_sidecar_files(&table_dir, &table_name, 1);

    backdate_dir(&table_dir, Duration::from_secs(3600));

    let dry = store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    for (name, _) in seg0_files.iter().chain(seg1_files.iter()) {
        assert!(
            !dry.orphans.iter().any(|o| o.ends_with(name.as_str())),
            "dimensions == 0: nothing about the current segment set is ever touched, so a \
             dry-run must never preview it as reclaimed: {dry:?}"
        );
    }
    assert_eq!(
        dry.bytes_reclaimed, 0,
        "dimensions == 0: no bytes are ever reclaimed from this row: {dry:?}"
    );

    let apply = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    let promoted = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        promoted.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "a dimensions == 0 row is still promoted, never reaped: {apply:?}"
    );
    for (name, _) in seg0_files.iter().chain(seg1_files.iter()) {
        assert!(
            table_dir.join(name).exists(),
            "dimensions == 0: apply must never delete stale sidecar '{name}' — \
             `rebuild_index_from_parquet` never calls `purge_segments` at all for this row"
        );
    }
    let segs = store
        .catalog()
        .list_index_segments(&table_name)
        .await
        .unwrap();
    assert_eq!(
        segs.len(),
        2,
        "both stale segments' catalog rows must survive untouched too: {segs:?}"
    );

    assert_eq!(dry.orphans, apply.orphans, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.bytes_reclaimed, apply.bytes_reclaimed,
        "{dry:?} vs {apply:?}"
    );
}

// ─── classifier oracle: the three [`jammi_db::store::ExpiredRowOutcome`]
//     shapes (Reap / Promote / Untouched) must never collapse into one
//     another — a mutation that made `classify_expired_row` always return
//     ONE outcome (e.g. `Untouched`) must go RED here, not merely be
//     inferred from a single-fixture test. ─────────────────────────────────

/// Three expired-lease `building` rows, each in a DIFFERENT classification
/// state, reconciled in the SAME admin pass: a torn row with no manifest
/// (`Reap`), a valid promotable row (`Promote`), and a row whose Parquet was
/// never even written (`Untouched`). Each must land in its own distinct
/// terminal shape — a classifier defect that collapsed any two of these
/// outcomes together is caught here directly, distinguishing it from a
/// pre-pass accounting bug the sibling tests above already cover.
#[tokio::test]
async fn classify_expired_row_never_collapses_reap_promote_and_untouched() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // Reap: valid Parquet, no manifest sidecar.
    let (reap_table, reap_parquet) =
        torn_building_row_fixture(&store, &catalog, "docs-classify-reap").await;

    // Promote: valid Parquet AND manifest sidecar.
    let (promote_table, promote_parquet) =
        promotable_building_row_fixture(&store, &catalog, "docs-classify-promote").await;

    // Untouched: a `building` row whose Parquet was never written at all —
    // the writer crashed before the very first byte landed.
    let untouched_info = store
        .create_table(
            "docs-classify-untouched",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            None,
        )
        .await
        .unwrap();
    let untouched_table = jammi_test_utils::abandon_building(&catalog, untouched_info).await;
    // Force the lease into the past directly (no bytes/table-dir exist yet
    // to backdate — `abandon_building` already left the lease expired).

    let apply = store
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    let reap_row = store
        .catalog()
        .get_result_table(&reap_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        reap_row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "Reap: {apply:?}"
    );
    assert!(
        !std::path::Path::new(&reap_parquet).exists(),
        "Reap must delete the torn row's Parquet"
    );

    let promote_row = store
        .catalog()
        .get_result_table(&promote_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        promote_row.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "Promote must never collapse to Reap or Untouched: {apply:?}"
    );
    assert!(
        std::path::Path::new(&promote_parquet).exists(),
        "Promote must never delete the row's Parquet"
    );

    let untouched_row = store
        .catalog()
        .get_result_table(&untouched_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        untouched_row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "Untouched: {apply:?}"
    );

    // The three rows are distinguishable by more than status alone: only
    // the `Reap` and `Promote` rows ever had bytes to account for.
    assert!(
        apply.orphans.iter().any(|o| o.ends_with(
            std::path::Path::new(&reap_parquet)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        )),
        "the Reap row's Parquet must be credited as reclaimed: {apply:?}"
    );
    assert!(
        !apply.orphans.iter().any(|o| o.ends_with(
            std::path::Path::new(&promote_parquet)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        )),
        "the Promote row's Parquet must never be credited as reclaimed: {apply:?}"
    );
}

/// esc-484 item "manifest vanished between classify and perform": a row
/// genuinely classified `Promote` (valid Parquet AND manifest sidecar both
/// present at classify time) whose manifest sidecar is deleted out from
/// under `reconcile_expired_building_row` in the exact window between that
/// classification and its own re-read must re-classify to `Reap` — never
/// abort the whole pass, and never a manifest-less promotion. Pinned
/// directly via the `reconcile_test_hooks` rendezvous (never merely
/// inferred): the manifest truly exists when `classify_expired_row` reads
/// it, and is deleted only once the pass has parked at the documented
/// re-read point.
#[cfg(feature = "test-hooks")]
#[tokio::test]
async fn manifest_vanished_between_classify_and_perform_re_classifies_to_reap() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let (table_name, parquet_local) =
        promotable_building_row_fixture(&store, &catalog, "docs-manifest-race").await;
    let manifest_path = parquet_local.replace(".parquet", ".materialization.json");
    assert!(
        std::path::Path::new(&manifest_path).exists(),
        "fixture sanity: the manifest sidecar must exist before the race"
    );

    let race = jammi_db::store::reconcile_test_hooks::arm_manifest_vanish_race(&table_name);
    let store_clone = store.clone();
    let handle = tokio::spawn(async move {
        store_clone
            .reconcile(ReconcileOptions {
                apply: true,
                grace: Duration::from_secs(3),
            })
            .await
    });

    race.wait_parked().await;
    assert!(
        race.is_parked(),
        "the recovery pass never reached the documented manifest re-read point"
    );
    std::fs::remove_file(&manifest_path).unwrap();
    race.release();

    let report = handle.await.unwrap().unwrap();

    let row = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "a manifest that vanished between classify and perform must re-classify to Reap, never \
         a manifest-less promotion: {report:?}"
    );
    assert!(
        !std::path::Path::new(&parquet_local).exists(),
        "the re-classified Reap arm must actually delete the Parquet"
    );
}

/// #484 design revision item 3: a Parquet that vanishes in the window
/// between `classify_expired_row`'s own `exists()` check and its single read
/// of the Parquet's bytes (`storage::reader::validate_and_count_parquet_rows`)
/// must re-classify to `Reap` — the identical outcome a torn/invalid Parquet
/// already gets — never propagate an object-store error that aborts the
/// WHOLE reconcile pass over one row's benign race. Pinned directly via the
/// `reconcile_test_hooks` rendezvous (never merely inferred): the Parquet
/// truly exists when `classify_expired_row`'s `exists()` check runs, and is
/// deleted only once the pass has parked at the documented re-read point.
#[cfg(feature = "test-hooks")]
#[tokio::test]
async fn parquet_vanished_during_the_classify_window_reaps_never_aborts_the_pass() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // A promotable fixture (valid Parquet AND manifest sidecar) so the
    // outcome the vanish forces (`Reap`) is unambiguously due to the race,
    // never a coincidental "no manifest" classification the row would have
    // reached anyway.
    let (table_name, parquet_local) =
        promotable_building_row_fixture(&store, &catalog, "docs-parquet-race").await;

    let race = jammi_db::store::reconcile_test_hooks::arm_parquet_vanish_race(&table_name);
    let store_clone = store.clone();
    let handle = tokio::spawn(async move {
        store_clone
            .reconcile(ReconcileOptions {
                apply: true,
                grace: Duration::from_secs(3),
            })
            .await
    });

    race.wait_parked().await;
    assert!(
        race.is_parked(),
        "the reconcile pass never reached the documented classify-window re-read point"
    );
    std::fs::remove_file(&parquet_local).unwrap();
    race.release();

    let report = handle
        .await
        .unwrap()
        .expect("the pass must complete, never abort, over a benign classify-window vanish");

    let row = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "a Parquet that vanished during the classify window must reap, never abort the pass or \
         yield a manifest-less promotion: {report:?}"
    );
}

/// esc-484 item (c): a catalog `index_segments` row whose `index_path` does
/// not even parse as a [`jammi_db::storage::StorageUrl`] is corruption —
/// `purge_segments` must hard-error rather than silently skip past it,
/// so the caller (here, `abort`) learns loudly that this table's segment
/// set could not be enumerated.
#[tokio::test]
async fn purge_segments_errors_on_an_unparseable_index_path() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let info = store
        .create_table(
            "docs-bad-segment-path",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            None,
        )
        .await
        .unwrap();
    // A raw catalog row naming a scheme `StorageUrl::parse` does not know —
    // the shape of on-disk/catalog corruption, never something a real
    // `append_segment` call could produce.
    let inserted = catalog
        .insert_index_segment(&info.cas(), 0, "bogus-scheme://wherever/seg0", 5)
        .await
        .unwrap();
    assert!(inserted, "fixture sanity: the segment row must land");

    let err = info.abort().await.unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("unparseable index_path"),
        "expected purge_segments' own corruption error, got: {message}"
    );
}

/// esc-484 item (c): a REAL `delete_if_exists` I/O failure (never a mere
/// 404) during `abort`'s byte cleanup must surface as ONE aggregated error
/// naming every key that failed to delete — not a silent `Ok(())` over a
/// half-cleaned row. Manufactured with an unwritable table directory
/// (`chmod 555`), which makes `unlink` fail with `EACCES` for every object
/// underneath, rather than any storage-failure test hook (none exists for
/// this path today).
#[tokio::test]
async fn abort_aggregates_a_real_delete_failure_into_one_error() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let info = create_building_embedding_with_parquet(&store, "docs-abort-fail", 3).await;
    info.append_segment(&built_segment_index(2)).await.unwrap();
    let table_name = info.table_name().to_string();
    let parquet_local = info
        .parquet_url()
        .as_str()
        .trim_start_matches("file://")
        .to_string();
    let table_dir = std::path::Path::new(&parquet_local)
        .parent()
        .unwrap()
        .to_path_buf();

    // Deleting a file requires WRITE permission on its containing
    // directory (POSIX `unlink` semantics) — strip it so every
    // `delete_if_exists` underneath fails with a real I/O error, never a
    // 404.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&table_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

    let result = info.abort().await;

    // Restore write permission unconditionally (before any assertion can
    // panic) so the tempdir's own Drop can clean up.
    std::fs::set_permissions(&table_dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    let err = result.unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("abort:") && message.contains("object delete(s) failed"),
        "expected abort's aggregated-failure error, got: {message}"
    );
    assert!(
        std::path::Path::new(&parquet_local).exists(),
        "a real delete failure must leave the Parquet in place for reconcile to retry"
    );
    let row = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "abort's own CAS still flips the row to failed — only the byte cleanup is incomplete"
    );
}

#[tokio::test]
async fn pre_pass_deleted_table_is_not_double_counted() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let (table_name, parquet_local) =
        torn_building_row_fixture(&store, &catalog, "docs-torn").await;
    let true_size = std::fs::metadata(&parquet_local).unwrap().len();
    let parquet_key = std::path::Path::new(&parquet_local)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // Dry-run FIRST — mutates nothing, so the state `apply=true` sees next
    // is bit-for-bit identical to what this pass already read.
    let dry = store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        std::path::Path::new(&parquet_local).exists(),
        "a dry-run must never delete the torn row's Parquet"
    );
    assert!(
        dry.orphans.iter().any(|o| o.ends_with(&parquet_key)),
        "a dry-run must PREVIEW the pre-pass's reap, not report it in no field: {dry:?}"
    );
    assert_eq!(
        dry.bytes_reclaimed, true_size,
        "a dry-run must preview the object's TRUE size, not 0: {dry:?}"
    );

    // Now apply, on the identical state.
    let apply = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    // The pre-pass reaped the row: it is `failed`, not still `building`.
    let row = store
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "{apply:?}"
    );
    // The pre-pass's own delete actually removed the bytes.
    assert!(
        !std::path::Path::new(&parquet_local).exists(),
        "the pre-pass must have deleted the torn row's Parquet"
    );

    // The RED oracle this replaces asserted the exact opposite of both of
    // the following — pasted verbatim from the pre-fix test body:
    //   assert!(
    //       !report.orphans.iter().any(|o| o.ends_with(&parquet_key)),
    //       "a pre-pass-deleted key must not double-count as this pass's own orphan: {report:?}"
    //   );
    //   assert_eq!(
    //       report.bytes_reclaimed, 0,
    //       "the pre-pass's own delete must not be double-counted in bytes_reclaimed: {report:?}"
    //   );
    // Reproduced at aa117dbf: apply=true reaped 1 object of `true_size`
    // bytes and reported `orphans: []`, `bytes_reclaimed: 0` — the reap was
    // real, the report was silent. The honest oracle: apply reports the
    // reaped key exactly once, with its true size — the SAME key and size
    // the preceding dry-run already reported on the identical state.
    assert!(
        apply.orphans.iter().any(|o| o.ends_with(&parquet_key)),
        "apply must report the key it actually reaped, exactly once: {apply:?}"
    );
    assert_eq!(apply.orphan_count, 1, "{apply:?}");
    assert_eq!(
        apply.bytes_reclaimed, true_size,
        "apply must report the TRUE bytes it reclaimed, not 0: {apply:?}"
    );
    assert_eq!(
        dry.orphans, apply.orphans,
        "dry-run and apply must agree on the identical state: {dry:?} vs {apply:?}"
    );
    assert_eq!(
        dry.bytes_reclaimed, apply.bytes_reclaimed,
        "{dry:?} vs {apply:?}"
    );
}

// ─── dry-run previews EXACTLY what the matching apply pass reclaims, field
//     by field, across every arm this pass has: the expired-building
//     pre-pass, the ready-row completeness arm, a healthy untouched table,
//     and an unattributed key (reported in neither arm, at any `apply`) ────

#[tokio::test]
async fn dry_run_previews_exactly_what_apply_reclaims() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());
    let ctx = SessionContext::new();

    // (1) Expired-building table with bytes (the pre-pass arm).
    let (_torn_table, torn_parquet) =
        torn_building_row_fixture(&store, &catalog, "docs-torn").await;

    // (2) A `ready` row with a missing manifest — the row->object
    // completeness arm; backdated so its now-unreferenced objects (once the
    // row fails) are unambiguously past grace, never merely `pending`.
    let missing_manifest_record = materialize_healthy_table(&store, &ctx, "docs-nomanifest").await;
    let mm_parquet_local = missing_manifest_record
        .parquet_path
        .trim_start_matches("file://")
        .to_string();
    let (mm_stem, _ext) = mm_parquet_local.rsplit_once('.').unwrap();
    std::fs::remove_file(format!("{mm_stem}.materialization.json")).unwrap();
    let mm_table_dir = std::path::Path::new(&mm_parquet_local).parent().unwrap();
    backdate_dir(mm_table_dir, Duration::from_secs(3600));

    // (3) A healthy table: must survive both passes untouched.
    let healthy_record = materialize_healthy_table(&store, &ctx, "docs-healthy").await;

    // (4) An unattributed key — never reported by any scoped pass, but this
    // test runs `reconcile_all` (admin), which does see it.
    let root = dir.path().join("jammi_db");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pre_layout.parquet"), b"stray-pre-layout").unwrap();
    backdate_dir(&root, Duration::from_secs(3600));

    // (5) An expired-lease `building` row with a valid Parquet AND its
    // manifest sidecar present — the PROMOTE state. Never reaped in EITHER
    // mode: recovery promotes it (apply); dry-run previews the same
    // classification and protects its objects. Backdated past grace, so the
    // fix under test — not a coincidental age match — is what keeps it out
    // of `orphans`.
    let (promote_table, promote_parquet) =
        promotable_building_row_fixture(&store, &catalog, "docs-promote").await;
    let promote_key = std::path::Path::new(&promote_parquet)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // (6) A torn `building` row (valid Parquet, no manifest) whose LEASE has
    // expired but whose bytes are YOUNGER than `grace` — the pre-pass's
    // CAS-licensed reap ignores object age entirely (only the general
    // object→row arm below consults `grace`), so this key must be an orphan
    // (never `pending`) in both modes.
    let (_young_torn_table, young_torn_parquet) =
        torn_building_row_fixture_young(&store, &catalog, "docs-young-torn").await;
    let young_torn_key = std::path::Path::new(&young_torn_parquet)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // Dry-run FIRST — mutates nothing.
    let dry = store
        .reconcile_all(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    // Sanity: the fixture actually exercises every arm (a vacuous "both
    // empty" pass would pass any field-by-field comparison trivially).
    assert!(
        dry.rows_failed
            .contains(&missing_manifest_record.table_name),
        "{dry:?}"
    );
    assert!(!dry.orphans.is_empty(), "{dry:?}");
    assert!(dry.bytes_reclaimed > 0, "{dry:?}");
    assert!(
        dry.unattributed
            .iter()
            .any(|u| u.ends_with("pre_layout.parquet")),
        "{dry:?}"
    );

    // The promote state: RED-first against be106f0c — before this fix, a
    // dry-run's pre-pass reported nothing special for a would-be-promoted
    // row (no protection), so its backdated Parquet fell through to the
    // ordinary age-gated orphan arm and was reported here. The honest
    // oracle: it must never appear, in either list.
    assert!(
        !dry.orphans.iter().any(|o| o.ends_with(&promote_key)),
        "a would-be-promoted row's Parquet must never be previewed as an orphan: {dry:?}"
    );
    assert!(
        !dry.pending.iter().any(|o| o.ends_with(&promote_key)),
        "a would-be-promoted row's Parquet must never be previewed as pending either: {dry:?}"
    );
    // The young-torn state: the pre-pass's CAS-licensed reap has no age
    // gate, so this key must be an orphan (never merely `pending`) even
    // though its bytes are fresher than `grace`.
    assert!(
        dry.orphans.iter().any(|o| o.ends_with(&young_torn_key)),
        "a CAS-licensed reap must report its key regardless of object age: {dry:?}"
    );
    assert!(
        !dry.pending.iter().any(|o| o.ends_with(&young_torn_key)),
        "the young torn row's key must never be `pending` — the pre-pass has no age gate: {dry:?}"
    );

    // Nothing was actually touched by the dry-run.
    assert!(std::path::Path::new(&torn_parquet).exists());
    assert!(std::path::Path::new(&mm_parquet_local).exists());
    assert!(std::path::Path::new(&promote_parquet).exists());
    assert!(std::path::Path::new(&young_torn_parquet).exists());
    assert!(root.join("pre_layout.parquet").exists());
    let healthy_still_ready = store
        .catalog()
        .get_result_table(&healthy_record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        healthy_still_ready.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string()
    );

    // Now apply, on the identical state.
    let apply = store
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    // RED-first: at aa117dbf, `dry.orphans`/`dry.bytes_reclaimed` under-
    // reported relative to `apply`'s on two counts at once — the pre-pass
    // silence this test's sibling above pins directly, and the ready-row
    // arm's `!apply` branch keeping a would-be-failed row's objects in
    // `still_ready` (so a dry-run never previewed them as reclaimable at
    // all, even though the matching `apply` pass deletes them). Every field
    // but `applied` must agree.
    assert_eq!(dry.scope, apply.scope);
    assert_eq!(dry.rows_failed, apply.rows_failed, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.rows_failed_count, apply.rows_failed_count,
        "{dry:?} vs {apply:?}"
    );
    assert_eq!(dry.orphans, apply.orphans, "{dry:?} vs {apply:?}");
    assert_eq!(dry.orphan_count, apply.orphan_count, "{dry:?} vs {apply:?}");
    assert_eq!(dry.pending, apply.pending, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.pending_count, apply.pending_count,
        "{dry:?} vs {apply:?}"
    );
    assert_eq!(dry.unattributed, apply.unattributed, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.unattributed_count, apply.unattributed_count,
        "{dry:?} vs {apply:?}"
    );
    assert_eq!(dry.damaged, apply.damaged, "{dry:?} vs {apply:?}");
    assert_eq!(
        dry.damaged_count, apply.damaged_count,
        "{dry:?} vs {apply:?}"
    );
    assert_eq!(
        dry.bytes_reclaimed, apply.bytes_reclaimed,
        "{dry:?} vs {apply:?}"
    );
    assert!(!dry.applied);
    assert!(apply.applied);

    // And apply actually did what it previewed: the reclaimed bytes are
    // gone, the healthy table and the unattributed key survive.
    assert!(!std::path::Path::new(&torn_parquet).exists());
    assert!(!std::path::Path::new(&mm_parquet_local).exists());
    assert!(root.join("pre_layout.parquet").exists());
    let healthy_after = store
        .catalog()
        .get_result_table(&healthy_record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        healthy_after.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string()
    );

    // The promote state: apply actually promoted the row (never reaped it),
    // and its Parquet — never deleted — is still on disk.
    let promote_after = store
        .catalog()
        .get_result_table(&promote_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        promote_after.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "recovery must promote a valid Parquet with its manifest present, not fail it"
    );
    assert!(
        std::path::Path::new(&promote_parquet).exists(),
        "a promoted row's Parquet must survive apply"
    );
    assert!(
        !apply.orphans.iter().any(|o| o.ends_with(&promote_key)),
        "apply must never report a promoted row's Parquet as an orphan: {apply:?}"
    );

    // The young-torn state: apply actually reaped it (its lease was expired,
    // regardless of its bytes' age).
    assert!(
        !std::path::Path::new(&young_torn_parquet).exists(),
        "apply must reap an expired-lease torn row regardless of its object age"
    );
    assert!(
        apply.orphans.iter().any(|o| o.ends_with(&young_torn_key)),
        "apply must report the young torn row's key as reclaimed: {apply:?}"
    );
}

// ─── S1: a healthy F32 indexed `ready` table survives `apply=true` ─────────

#[tokio::test]
async fn healthy_table_survives_reconcile_apply() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store, &ctx, "docs").await;

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();

    assert!(report.rows_failed.is_empty(), "{report:?}");
    assert!(report.orphans.is_empty(), "{report:?}");
    assert!(report.unattributed.is_empty(), "{report:?}");
    assert_eq!(report.bytes_reclaimed, 0);

    // The table is still there, byte-for-byte reachable.
    let still_there = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap();
    assert!(
        still_there.is_some(),
        "reconcile must not touch a live table"
    );
}

// ─── apply=false mutates nothing, even with a missing object ───────────────

#[tokio::test]
async fn apply_false_never_mutates() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store, &ctx, "docs").await;

    // Delete the Parquet out from under the row directly (never through the
    // engine) to manufacture a missing-required-object condition.
    std::fs::remove_file(record.parquet_path.trim_start_matches("file://")).ok();

    let report = store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(0),
        })
        .await
        .unwrap();
    assert!(
        report.rows_failed.contains(&record.table_name),
        "a missing required object must be REPORTED even under apply=false: {report:?}"
    );

    // But nothing was mutated: the row is still `ready`.
    let still_ready = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        still_ready.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "apply=false must not flip the row"
    );
}

// ─── missing required object -> `ready -> failed`, then its objects are
//     orphans (reaped only past grace, under `apply`) ────────────────────

#[tokio::test]
async fn missing_object_fails_the_row_and_is_reaped_past_grace() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store, &ctx, "docs").await;

    // Remove ONE required sidecar sibling (not the Parquet itself) so the
    // Parquet object survives to become the orphan candidate reconcile then
    // reaps.
    let segs = store
        .catalog()
        .list_index_segments(&record.table_name)
        .await
        .unwrap();
    assert_eq!(segs.len(), 1, "5 rows fit in one segment");
    let idx_path = segs[0].index_path.trim_start_matches("file://");
    let (stem, _ext) = idx_path.rsplit_once('.').unwrap();
    let usearch_path = format!("{stem}.usearch");
    std::fs::remove_file(&usearch_path).unwrap();
    // Backdate every OTHER surviving object of this table so it is already
    // past a short grace window when reconcile runs — no real-time sleep
    // needed to prove "past grace" reclaim.
    let table_dir = std::path::Path::new(idx_path).parent().unwrap();
    backdate_dir(table_dir, Duration::from_secs(3600));

    let report = store
        .clone()
        .with_lease_intervals(short_lease())
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        report.rows_failed.contains(&record.table_name),
        "missing sidecar must fail the row: {report:?}"
    );
    // At grace=0 every orphan candidate (now including the Parquet + rowmap +
    // manifest siblings, since the row is no longer `ready`) is reclaimed.
    assert!(
        !report.orphans.is_empty(),
        "the failed row's now-unreferenced objects must be orphan candidates: {report:?}"
    );
    assert!(report.bytes_reclaimed > 0);

    let row = store
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string()
    );
}

// ─── a young orphan is `pending`, never deleted, at any `apply` ───────────

#[tokio::test]
async fn young_orphan_is_pending_not_deleted() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();

    // A stray object under the `_global` segment with no catalog row at all
    // (a losing writer's abandoned bytes) — attributable, but unreferenced.
    let stray = dir.path().join("jammi_db").join("_global");
    std::fs::create_dir_all(&stray).unwrap();
    std::fs::write(stray.join("orphan_table.parquet"), b"stray-bytes").unwrap();

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(
        report
            .pending
            .iter()
            .any(|p| p.ends_with("orphan_table.parquet")),
        "a fresh orphan must be `pending` under a long grace: {report:?}"
    );
    assert!(report.orphans.is_empty());
    assert!(stray.join("orphan_table.parquet").exists());
}

// ─── an unattributed key survives even at the SHORTEST valid grace under
//     `apply=true` (unattributed is skipped before the age gate at all,
//     so this holds independent of the object's age) — reported ONLY by
//     the admin cross-tenant pass (a scoped pass, even an
//     unscoped/GLOBAL one, reports NOTHING it cannot attribute to its own
//     prefix; an unattributed key is store-wide by definition) ───────────

#[tokio::test]
async fn unattributed_key_never_deleted_regardless_of_grace() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    // A pre-layout / garbage key directly at the root — its own first path
    // segment does not parse through `TenantSegment::parse` at all.
    let root = dir.path().join("jammi_db");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pre_layout_table.parquet"), b"ancient-bytes").unwrap();
    backdate_dir(&root, Duration::from_secs(3600));

    let report = store
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        report
            .unattributed
            .contains(&"pre_layout_table.parquet".to_string()),
        "{report:?}"
    );
    assert_eq!(report.unattributed_count, 1);
    assert!(report.orphans.is_empty());
    assert!(root.join("pre_layout_table.parquet").exists());
}

/// RED first: the SAME stray key as above, but seen through the
/// SCOPED arm (`ResultStore::reconcile`, even on an unscoped/GLOBAL store —
/// scoped is scoped regardless of which tenant) — must report NOTHING for
/// it. Before the fix this failed: a scoped pass listed every unattributed
/// key store-wide.
#[tokio::test]
async fn scoped_reconcile_never_reports_unattributed_keys() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let root = dir.path().join("jammi_db");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("pre_layout_table.parquet"), b"ancient-bytes").unwrap();
    backdate_dir(&root, Duration::from_secs(3600));

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(
        report.unattributed.is_empty(),
        "a scoped pass must report NO unattributed entries: {report:?}"
    );
    assert_eq!(report.unattributed_count, 0, "{report:?}");
    // Never deleted either way — out of scope, not merely "not old enough".
    assert!(root.join("pre_layout_table.parquet").exists());
}

// ─── apply=true requires grace >= lease.duration ──────────────────────────

#[tokio::test]
async fn apply_requires_grace_at_least_the_lease_duration() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), catalog, AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let err = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_millis(1),
        })
        .await
        .unwrap_err();
    assert!(
        matches!(err, jammi_db::error::JammiError::Config(_)),
        "a too-short grace under apply=true must be a typed Config refusal, got: {err:?}"
    );

    // apply=false tolerates ANY grace.
    store
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_millis(1),
        })
        .await
        .unwrap();
}

// ─── two-tenant scope: a tenant's own reconcile never touches / reports
//     another tenant's rows or stray objects; reconcile_all covers both ───

#[tokio::test]
async fn tenant_scoped_reconcile_never_touches_another_tenants_prefix() {
    let dir = tempdir().unwrap();
    let base_catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let tenant_a = fresh_tenant();
    let tenant_b = fresh_tenant();
    let catalog_a = Arc::new(base_catalog.pinned_to_tenant(Some(tenant_a)));
    let catalog_b = Arc::new(base_catalog.pinned_to_tenant(Some(tenant_b)));

    let store_a = ResultStore::new(
        dir.path(),
        Arc::clone(&catalog_a),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());
    let store_b = ResultStore::new(
        dir.path(),
        Arc::clone(&catalog_b),
        AnnIndexConfig::default(),
    )
    .unwrap();
    let ctx = SessionContext::new();
    let record_a = materialize_healthy_table(&store_a, &ctx, "docs-a").await;
    let record_b = materialize_healthy_table(&store_b, &ctx, "docs-b").await;

    // A stray object under tenant B's own segment — B's problem, not A's.
    let stray_dir = dir.path().join("jammi_db").join(tenant_b.to_string());
    std::fs::create_dir_all(&stray_dir).unwrap();
    std::fs::write(stray_dir.join("stray.parquet"), b"stray").unwrap();
    backdate_dir(&stray_dir, Duration::from_secs(3600));

    // Tenant A's own reconcile: sees nothing of B's, reports nothing of B's,
    // and never deletes B's stray (it is out of A's scope, not A's orphan).
    let report_a = store_a
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(report_a.rows_failed.is_empty(), "{report_a:?}");
    assert!(report_a.orphans.is_empty(), "{report_a:?}");
    assert!(stray_dir.join("stray.parquet").exists());

    // Both tables (A's and B's) are still intact after A's own pass.
    assert!(store_a
        .catalog()
        .get_result_table(&record_a.table_name)
        .await
        .unwrap()
        .is_some());
    assert!(store_b
        .catalog()
        .get_result_table(&record_b.table_name)
        .await
        .unwrap()
        .is_some());

    // reconcile_all (admin) reaps ONLY the stray, leaving both tenants'
    // live tables byte-identical.
    let report_all = store_a
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();
    assert!(report_all.rows_failed.is_empty(), "{report_all:?}");
    assert!(
        report_all
            .orphans
            .iter()
            .any(|o| o.ends_with("stray.parquet")),
        "{report_all:?}"
    );
    assert!(!stray_dir.join("stray.parquet").exists());
    assert!(store_a
        .catalog()
        .get_result_table(&record_a.table_name)
        .await
        .unwrap()
        .is_some());
    assert!(store_b
        .catalog()
        .get_result_table(&record_b.table_name)
        .await
        .unwrap()
        .is_some());
}

// ─── a scoped pass reports NOTHING it cannot act on: a GLOBAL ready row
//     with a missing required object is invisible to a tenant-scoped
//     dry-run AND apply alike — only an admin (`all=true`) pass ever
//     touches it. RED before the fix: `list_result_tables_by_status` (an
//     ordinary READ — GLOBAL visible to every tenant, like any other read on
//     this table) fed straight into the row->object CAS below it, so a
//     tenant-scoped dry-run REPORTED the GLOBAL row in `rows_failed` while
//     the matching `apply` pass's `fail_ready_result_table` CAS (`Strict` on
//     the caller's own tenant) missed it entirely — dry-run and apply
//     disagreed on the identical state. ─────────────────────────────────────

#[tokio::test]
async fn scoped_pass_reports_nothing_for_a_global_row_it_cannot_act_on() {
    let dir = tempdir().unwrap();
    let base_catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let tenant_b = fresh_tenant();
    let catalog_b = Arc::new(base_catalog.pinned_to_tenant(Some(tenant_b)));

    let store_global = ResultStore::new(
        dir.path(),
        Arc::clone(&base_catalog),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());
    let store_b = ResultStore::new(
        dir.path(),
        Arc::clone(&catalog_b),
        AnnIndexConfig::default(),
    )
    .unwrap()
    .with_lease_intervals(short_lease());
    let ctx = SessionContext::new();
    let record = materialize_healthy_table(&store_global, &ctx, "docs-global").await;

    // Remove the GLOBAL row's `.materialization.json` sidecar so it fails
    // `required_row_objects_present` — a missing required object, exactly
    // the condition that would (incorrectly) drive a scoped dry-run to
    // report it.
    let parquet_local = record.parquet_path.trim_start_matches("file://");
    let (stem, _ext) = parquet_local.rsplit_once('.').unwrap();
    let mat_path = format!("{stem}.materialization.json");
    assert!(
        std::path::Path::new(&mat_path).exists(),
        "fixture must actually produce a materialization sidecar to remove"
    );
    std::fs::remove_file(&mat_path).unwrap();

    // Tenant B's dry-run: reports NOTHING for the GLOBAL row.
    let dry = store_b
        .reconcile(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(dry.rows_failed.is_empty(), "{dry:?}");
    assert_eq!(dry.rows_failed_count, 0, "{dry:?}");

    // Tenant B's apply: agrees with its own dry-run on the same state.
    let apply = store_b
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(apply.rows_failed.is_empty(), "{apply:?}");
    assert_eq!(apply.rows_failed_count, 0, "{apply:?}");

    // The GLOBAL row stayed `ready` through both of tenant B's passes.
    let row = store_global
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status,
        jammi_db::catalog::status::ResultTableStatus::Ready.to_string(),
        "a scoped pass must never flip a row it cannot even report on"
    );

    // An admin (`all=true`) pass sees and reports it in BOTH modes.
    let admin_dry = store_b
        .reconcile_all(ReconcileOptions {
            apply: false,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(
        admin_dry.rows_failed.contains(&record.table_name),
        "{admin_dry:?}"
    );

    let admin_apply = store_b
        .reconcile_all(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3600),
        })
        .await
        .unwrap();
    assert!(
        admin_apply.rows_failed.contains(&record.table_name),
        "{admin_apply:?}"
    );
    let row_after = store_global
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row_after.status,
        jammi_db::catalog::status::ResultTableStatus::Failed.to_string(),
        "the admin apply pass must actually act on the row it reported"
    );
}

// ─── artifact arm: a running job's checkpoints survive; a canonical-UUID
//     job id never lands in `unattributed` ─────────────────────────────────

#[tokio::test]
async fn running_jobs_artifact_prefix_survives_and_is_never_unattributed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();

    catalog
        .register_model(RegisterModelParams {
            model_id: "base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();

    let job_id = Uuid::new_v4().to_string();
    catalog
        .create_training_job(CreateTrainingJobParams {
            job_id: &job_id,
            base_model_id: "base::1",
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    catalog
        .claim_next_training_job("worker-1", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("the freshly queued job is claimable");

    let bundle = vec![(
        "adapter.safetensors".to_string(),
        bytes::Bytes::from_static(b"weights"),
    )];
    store
        .artifact_store()
        .put_artifact(None, &[&job_id, "worker-1", "0"], &bundle)
        .await
        .unwrap();

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(30),
        })
        .await
        .unwrap();
    assert!(
        report.unattributed.is_empty(),
        "a canonical-UUID job id under a running job must never be unattributed: {report:?}"
    );
    assert!(
        report.orphans.is_empty(),
        "a running job's published bytes must survive reconcile: {report:?}"
    );

    // The bytes are still there.
    assert!(dir
        .path()
        .join("jammi_db")
        .join("models")
        .join("_global")
        .join(&job_id)
        .join("worker-1")
        .join("0")
        .join("adapter.safetensors")
        .exists());
}

// ─── RED first: a `models` row names a prefix whose `manifest.json`
//     is absent — the row is still live, so the prefix is NEVER reclaimed
//     through the orphan arm, at any grace or `apply`; it is reported as
//     `damaged` instead (row present, manifest absent) ────────────────────

#[tokio::test]
async fn models_row_prefix_with_no_manifest_is_damaged_never_reclaimed() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default())
        .unwrap()
        .with_lease_intervals(short_lease());

    let job_id = Uuid::new_v4().to_string();
    let bundle = vec![(
        "adapter.safetensors".to_string(),
        bytes::Bytes::from_static(b"weights"),
    )];
    let prefix_url = store
        .artifact_store()
        .put_artifact(None, &[&job_id], &bundle)
        .await
        .unwrap();

    // A `models` row names exactly this prefix as its winning attempt.
    catalog
        .register_model(RegisterModelParams {
            model_id: "damaged-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: Some(prefix_url.as_str()),
            config_json: None,
        })
        .await
        .unwrap();

    // Torn write / corrupted publish: the manifest is gone, the weights
    // survive. Backdate everything so it is well past a short grace.
    let prefix_dir = dir
        .path()
        .join("jammi_db")
        .join("models")
        .join("_global")
        .join(&job_id);
    std::fs::remove_file(prefix_dir.join("manifest.json")).unwrap();
    backdate_dir(&prefix_dir, Duration::from_secs(3600));

    let report = store
        .reconcile(ReconcileOptions {
            apply: true,
            grace: Duration::from_secs(3),
        })
        .await
        .unwrap();

    assert!(
        report
            .damaged
            .iter()
            .any(|d| d.ends_with("adapter.safetensors")),
        "a models-row-named prefix with no manifest must be reported damaged: {report:?}"
    );
    assert_eq!(report.damaged_count, 1, "{report:?}");
    assert!(
        report
            .orphans
            .iter()
            .all(|o| !o.ends_with("adapter.safetensors")),
        "damaged bytes must never fall through to the orphan arm: {report:?}"
    );
    assert!(
        prefix_dir.join("adapter.safetensors").exists(),
        "damaged bytes are never reclaimed, even past grace under apply=true"
    );
}
