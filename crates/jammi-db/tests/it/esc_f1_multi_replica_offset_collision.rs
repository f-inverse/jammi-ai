//! Escape row esc-F1 (issue-triage symptom_spec; PLAN-F §3/F4/G5/H5,
//! `<eval-verdict>` F15(i)): **multi-replica offset collision**.
//!
//! Symptom: two engine replicas (or, in-process, two independent
//! [`Publisher`] instances wired to the SAME catalog backend and topic --
//! the shape two OS processes sharing one Postgres database would take)
//! publish concurrently. Before this fix, [`Publisher`] assigned offsets
//! from a per-process `AtomicU64` counter lazily seeded from `MAX(_offset)`
//! on the backing table -- read ONCE, then incremented locally forever
//! after. Two such counters seeded against the SAME empty table both start
//! at `0` and increment independently, **oblivious to each other**: they
//! assign the identical offset sequence `0, 1, 2, …` to two disjoint sets of
//! rows, colliding on the backing table's `(_offset, _row_idx)` composite
//! primary key.
//!
//! Control: a SINGLE `Publisher` publishing the same total row count assigns
//! a gap-free, collision-free offset sequence (already covered by
//! `trigger.rs`'s existing single-publisher tests).
//!
//! Fix (this commit): the offset is assigned by ONE row-locked
//! `UPDATE topics SET next_offset = … RETURNING` statement inside the SAME
//! transaction as the row insert (`topics.next_offset`, migration 028) --
//! the row lock on the `topics` catalog row is the cross-instance mutual
//! exclusion, so two `Publisher`s (in-process OR cross-process on Postgres)
//! serialize on it and never compute the same offset twice.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use jammi_db::catalog::backend::BackendImpl;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::catalog::Catalog;
use jammi_db::source::mutable::MutableTableRegistry;
use jammi_db::store::mutable::sqlite::SqliteMutableBackend;
use jammi_db::store::mutable::MutableBackend;
use jammi_db::tenant_scope::TenantBinding;
use jammi_db::trigger::{
    InMemoryBroker, Offset, Predicate, Publisher, Subscriber, TopicDefinition, TopicId,
    TriggerBroker,
};

fn topic_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "marker",
        DataType::Int64,
        false,
    )]))
}

fn batch_of(marker: i64) -> RecordBatch {
    RecordBatch::try_new(
        topic_schema(),
        vec![Arc::new(Int64Array::from(vec![marker]))],
    )
    .unwrap()
}

/// Publish `n` single-row batches through `publisher`, tagging each row with
/// `marker` so the oracle can attribute rows back to whichever `Publisher`
/// "replica" wrote them. Returns the assigned offsets in publish order,
/// ignoring publish errors (a collision may surface as either an `Err` from
/// a primary-key violation OR a silently duplicated offset value --
/// tolerating both keeps the oracle meaningful regardless of which shape a
/// given backend's constraint enforcement takes).
async fn publish_n(
    publisher: &Publisher,
    topic: &TopicDefinition,
    n: usize,
    marker: i64,
) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(n);
    for _ in 0..n {
        if let Ok(off) = publisher
            .publish_scoped(topic, None, batch_of(marker))
            .await
        {
            offsets.push(off.value());
        }
    }
    offsets
}

#[tokio::test]
async fn two_replica_publishers_assign_gap_free_offsets() {
    let dir = tempfile::tempdir().unwrap();
    let sqlite = SqliteBackend::open(&dir.path().join("catalog.db"))
        .await
        .unwrap();
    let backend_impl = BackendImpl::Sqlite(sqlite);
    backend_impl.migrate().await.unwrap();

    let tenant_binding = TenantBinding::unscoped();
    let catalog = Arc::new(Catalog::from_backend_with_tenant(
        backend_impl,
        Some(tenant_binding.clone()),
    ));
    let backend_arc = catalog.backend_arc();
    let mutable_backend: Arc<dyn MutableBackend> =
        Arc::new(SqliteMutableBackend::new(Arc::clone(&backend_arc)));
    let registry = Arc::new(MutableTableRegistry::new(
        Arc::clone(&catalog),
        mutable_backend,
        tenant_binding,
    ));
    let broker: Arc<dyn TriggerBroker> = Arc::new(InMemoryBroker::new());
    let topic_repo = TopicRepo::new(Arc::clone(&catalog), Arc::clone(&registry));

    let topic = TopicDefinition {
        id: TopicId::new(),
        name: "esc_f1.multi_replica".to_string(),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    };
    broker.register_topic(&topic).await.unwrap();
    topic_repo.register_topic(&topic).await.unwrap();

    // Two INDEPENDENT `Publisher` instances over the SAME backend/registry/
    // broker/topic -- the in-process stand-in for two engine replicas on one
    // shared Postgres database. Each owns its own internal state; nothing
    // synchronizes them except the catalog itself.
    let publisher_a = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );
    let publisher_b = Publisher::new(
        Arc::clone(&broker),
        Arc::clone(&backend_arc),
        Arc::clone(&registry),
    );

    const N: usize = 25;
    let (offsets_a, offsets_b) = tokio::join!(
        publish_n(&publisher_a, &topic, N, 1),
        publish_n(&publisher_b, &topic, N, 2),
    );

    // Every publish must have succeeded -- a collision that surfaces as a
    // constraint-violation `Err` is itself a symptom, not a graceful
    // degradation.
    assert_eq!(
        offsets_a.len(),
        N,
        "publisher A must have every publish succeed; got {} of {N} (errors indicate PK collisions)",
        offsets_a.len()
    );
    assert_eq!(
        offsets_b.len(),
        N,
        "publisher B must have every publish succeed; got {} of {N} (errors indicate PK collisions)",
        offsets_b.len()
    );

    // The oracle: replay the topic's full history and check the offset SET
    // is a gap-free permutation of `0..2N` -- i.e. every one of the 2N rows
    // landed at a UNIQUE offset, regardless of which "replica" wrote it.
    let subscriber = Subscriber::new(Arc::clone(&broker), Arc::clone(&registry));
    let from = Offset::new(0, chrono::Utc::now());
    let drained = subscriber
        .replay_only(&topic, Predicate::match_all(), Some(from))
        .await
        .unwrap();
    let mut seen_offsets: Vec<u64> = drained.iter().map(|d| d.offset.value()).collect();
    seen_offsets.sort_unstable();
    let expected: Vec<u64> = (0..(2 * N) as u64).collect();
    assert_eq!(
        seen_offsets,
        expected,
        "two replicas publishing {N} rows each must assign a gap-free permutation of 0..{} \
         -- got {seen_offsets:?} (duplicates/gaps indicate the multi-replica offset collision)",
        2 * N
    );

    // A stronger form of the same property: no offset value repeats across
    // the combined publish sets from A and B.
    let mut combined: Vec<u64> = offsets_a.iter().chain(offsets_b.iter()).copied().collect();
    combined.sort_unstable();
    let mut deduped = combined.clone();
    deduped.dedup();
    assert_eq!(
        combined.len(),
        deduped.len(),
        "no offset may be assigned to both a publisher-A row and a publisher-B row; \
         combined offsets: {combined:?}"
    );
}
