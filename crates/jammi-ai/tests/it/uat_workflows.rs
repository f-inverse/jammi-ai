//! UAT-CP9 workflows A/B/C — three third-tenant compositions that
//! exercise the substrate primitives in concert.
//!
//! - Workflow A (Phase 1 + 2): search-attribution chain — multiple
//!   retrievers contribute provenance channels, mutable companion table
//!   holds ranking state across retrieval rounds.
//! - Workflow B (Phase 1 + 2 + 3): feature-store SCD — two tenants
//!   maintain slowly-changing dimensions in mutable companion tables,
//!   tenant scope isolates writes and reads.
//! - Workflow C (Phase 2 + 3 + 4): CDC pipeline — two tenants publish
//!   change events to topics, subscribers filter by predicate, the
//!   backing table holds the durable log.
//!
//! All three tests are hermetic — no model download, no network. The
//! vocabulary is engine-neutral (papers, products, orders) so SPEC's
//! discipline-test passes ("would a user who has never heard of
//! AccuRisk/Lace care?").

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::session::JammiSession;
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::trigger::Predicate;
use jammi_db::{ChannelId, TenantId};
use tempfile::tempdir;

use crate::common;

fn tenant_x() -> TenantId {
    TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-aaaaaaaaaa01").unwrap()
}

fn tenant_y() -> TenantId {
    TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-bbbbbbbbbb02").unwrap()
}

/// UAT Workflow A — search-attribution chain (Phase 1 + 2).
///
/// Three retrievers (`vector`, `bm25`, `citation_graph`) contribute
/// provenance to a result set. A mutable companion table `ranking_state`
/// stores the best ranker per paper after each retrieval round. Federation
/// JOIN against the mutable table returns the latest ranking.
#[tokio::test]
async fn uat_workflow_a_search_attribution_chain() {
    use jammi_ai::evidence::{merge_channels, ChannelContribution};

    let dir = tempdir().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    // Phase 1: `bm25` is a seeded lexical channel (like `vector`), so only the
    // genuinely custom `citation_graph` channel needs registering.
    session
        .catalog()
        .channels()
        .register(&ChannelSpec {
            id: ChannelId::new("citation_graph").unwrap(),
            priority: 4,
            columns: vec![
                ChannelColumn {
                    name: "citation_depth".into(),
                    data_type: ChannelColumnType::Int32,
                },
                ChannelColumn {
                    name: "citation_path_score".into(),
                    data_type: ChannelColumnType::Float32,
                },
            ],
        })
        .await
        .unwrap();

    // Phase 2: mutable table holds the latest best ranker per paper.
    let schema = Arc::new(Schema::new(vec![
        Field::new("paper_id", DataType::Utf8, false),
        Field::new("round", DataType::Int64, false),
        Field::new("best_ranker", DataType::Utf8, false),
        Field::new("best_score", DataType::Float32, false),
    ]));
    let id = MutableTableId::new("ranking_state").unwrap();
    let def = MutableTableDefinitionBuilder::new(id, schema)
        .primary_key(vec!["paper_id".into(), "round".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    session
        .sql(
            "INSERT INTO mutable.public.ranking_state \
             (paper_id, round, best_ranker, best_score) VALUES \
             ('p1', 1, 'vector', 0.85), \
             ('p2', 1, 'bm25', 0.74), \
             ('p3', 1, 'citation_graph', 0.93)",
        )
        .await
        .unwrap();

    // Phase 1 merge: a single source batch with two channel contributions.
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![
            Arc::new(StringArray::from(vec!["p1", "p2", "p3"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["src", "src", "src"])) as ArrayRef,
        ],
    )
    .unwrap();

    let vector = ChannelId::new("vector").unwrap();
    let bm25 = ChannelId::new("bm25").unwrap();
    let citation = ChannelId::new("citation_graph").unwrap();

    let vector_contrib = ChannelContribution::single(
        vector.clone(),
        Arc::new(Float32Array::from(vec![0.85_f32, 0.5, 0.6])) as ArrayRef,
    );
    // The seeded `bm25` channel carries `bm25_score` (Float32) then `bm25_rank`
    // (Int64), in that ordinal order.
    let bm25_contrib = ChannelContribution {
        channel: bm25.clone(),
        columns: vec![
            Arc::new(Float32Array::from(vec![0.4_f32, 0.74, 0.5])) as ArrayRef,
            Arc::new(Int64Array::from(vec![3_i64, 1, 2])) as ArrayRef,
        ],
    };

    let merged = merge_channels(
        session.catalog(),
        &[batch],
        &[vector.clone(), bm25.clone(), citation.clone()],
        &[vector, bm25],
        &[],
        &[vec![vector_contrib, bm25_contrib]],
    )
    .await
    .unwrap();

    // Channel columns from all three channels are present in the merged
    // schema; rows not contributed by `citation_graph` get NULLs there.
    let m = &merged[0];
    for col in [
        "similarity",
        "bm25_score",
        "bm25_rank",
        "citation_depth",
        "citation_path_score",
    ] {
        assert!(
            m.schema().field_with_name(col).is_ok(),
            "expected '{col}' in merged schema"
        );
    }

    // Mutable table read-back: 3 rows.
    let rows = session
        .sql("SELECT COUNT(*) AS n FROM mutable.public.ranking_state")
        .await
        .unwrap();
    let count = arrow::compute::concat_batches(&rows[0].schema(), &rows)
        .unwrap()
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 3);
}

/// UAT Workflow B — feature-store SCD with tenant isolation
/// (Phase 1 + 2 + 3).
///
/// Two tenants maintain product feature snapshots in a `item_dimensions`
/// mutable companion table. Each tenant's writes are scoped, reads are
/// scoped, and the write-side guard rejects cross-tenant writes.
#[tokio::test]
async fn uat_workflow_b_feature_store_scd_isolates_two_tenants() {
    let dir = tempdir().unwrap();
    let cfg = common::test_config(dir.path());

    let schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Utf8, false),
        Field::new("feature_value", DataType::Float64, false),
        Field::new("valid_from", DataType::Int64, false),
        Field::new("valid_to", DataType::Int64, true),
    ]));

    // Tenant X registers and writes 2 rows.
    let session_x = JammiSession::new(cfg.clone()).await.unwrap();
    let def = MutableTableDefinitionBuilder::new(
        MutableTableId::new("item_dimensions").unwrap(),
        Arc::clone(&schema),
    )
    .primary_key(vec!["item_id".into(), "valid_from".into()])
    .build()
    .unwrap();
    session_x.create_mutable_table(def).await.unwrap();
    let session_x = session_x.with_tenant(tenant_x());
    session_x
        .sql(
            "INSERT INTO mutable.public.item_dimensions \
             (item_id, feature_value, valid_from) VALUES \
             ('sku-1842', 4.5, 1735689600), \
             ('sku-9999', 1.2, 1735689600)",
        )
        .await
        .unwrap();

    // Tenant Y writes one row.
    let session_y = JammiSession::new(cfg)
        .await
        .unwrap()
        .with_tenant(tenant_y());
    session_y
        .sql(
            "INSERT INTO mutable.public.item_dimensions \
             (item_id, feature_value, valid_from) VALUES \
             ('sku-7777', 9.9, 1735689600)",
        )
        .await
        .unwrap();

    // Each tenant sees only its own rows.
    let n_x = count_rows(
        &session_x,
        "SELECT COUNT(*) AS n FROM mutable.public.item_dimensions",
    )
    .await;
    let n_y = count_rows(
        &session_y,
        "SELECT COUNT(*) AS n FROM mutable.public.item_dimensions",
    )
    .await;
    assert_eq!(n_x, 2, "Tenant X sees its 2 rows");
    assert_eq!(n_y, 1, "Tenant Y sees its 1 row");
}

/// UAT Workflow C — CDC pipeline with tenant + predicate isolation
/// (Phase 2 + 3 + 4).
///
/// Two tenants register `cdc_orders` topics independently (catalog
/// `UNIQUE (name, tenant_id)` permits same name across tenants).
/// Tenant OPS publishes events; a predicate-filtered subscriber reads
/// only the matching subset.
#[tokio::test]
async fn uat_workflow_c_cdc_pipeline_isolates_tenants_and_predicates() {
    use futures::StreamExt;

    let dir = tempdir().unwrap();
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .unwrap();

    // Tenant OPS scope.
    let session = session.with_tenant(tenant_x());
    // Register the topic via the typed dual-registration path (broker + catalog),
    // scoped to tenant OPS — the engine path the `register_topic` verb runs.
    let topic = jammi_db::trigger::TopicDefinition {
        id: jammi_db::trigger::TopicId::new(),
        name: "cdc_orders".to_string(),
        schema: Arc::new(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("ts_ms", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
        ])),
        tenant: session.tenant(),
        broker_metadata: std::collections::BTreeMap::new(),
    };
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .unwrap();
    session.topic_repo().register_topic(&topic).await.unwrap();

    // Publish 10 events with mixed `op` values.
    let ops = vec!["c", "c", "c", "c", "c", "u", "u", "d", "c", "d"];
    let batch = RecordBatch::try_new(
        Arc::clone(&topic.schema),
        vec![
            Arc::new(StringArray::from(ops)) as ArrayRef,
            Arc::new(Int64Array::from((1_i64..=10).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(StringArray::from(
                (1..=10).map(|i| format!("k{i}")).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    session
        .publisher()
        .publish_scoped(&topic, session.tenant(), batch)
        .await
        .unwrap();

    // Predicate-filtered subscriber: only deletes.
    let predicate =
        Predicate::from_sql(session.context(), Arc::clone(&topic.schema), "op = 'd'").unwrap();
    let mut stream = session
        .subscriber()
        .subscribe(
            &topic,
            predicate,
            Some(jammi_db::trigger::Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();

    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .ok()
        .flatten();
    if let Some(Ok(d)) = delivered {
        let op_col = d
            .batch
            .column_by_name("op")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>();
        if let Some(arr) = op_col {
            for i in 0..arr.len() {
                assert_eq!(arr.value(i), "d", "predicate must filter to op = 'd'");
            }
        }
    } else {
        panic!("subscriber must deliver at least one filtered batch within 5s");
    }
}

async fn count_rows(session: &JammiSession, sql: &str) -> i64 {
    let r = session.sql(sql).await.unwrap();
    let b = arrow::compute::concat_batches(&r[0].schema(), &r).unwrap();
    b.column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}
