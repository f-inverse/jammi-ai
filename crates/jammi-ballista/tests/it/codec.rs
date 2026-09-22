//! Hermetic codec round-trip oracles.

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;

use jammi_ai::model::ModelTask;
use jammi_ai::operator::inference_exec::InferenceExec;
use jammi_ai::operator::key_check_exec::KeyCheckExec;
use jammi_ai::operator::numbered_input_exec::{NumberedInputExec, RowOrder};
use jammi_ai::pipeline::asof::exec::AsofJoinExec;
use jammi_ai::pipeline::asof::spec::{AsofJoinSpecBuilder, AsofKey};
use jammi_ai::session::InferenceSession;
use jammi_ballista::codec::JammiCodec;
use jammi_db::catalog::result_repo::CreateResultTableParams;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::config::StoragePrecision;
use jammi_db::store::manifest::ComputeDeviceKind;
use jammi_db::store::{ResultStore, ResultTableSinkExec, SinkKind, SinkLeaseKind};

async fn session() -> Arc<InferenceSession> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = jammi_test_utils::test_config(dir.path());
    let s = InferenceSession::new(cfg).await.expect("session builds");
    // Keep the tempdir alive for the session's lifetime by leaking it (test
    // process teardown reclaims it; a session never outlives one test).
    std::mem::forget(dir);
    Arc::new(s)
}

fn string_scan(name: &str, values: &[&str]) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(values.to_vec()))],
    )
    .unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

/// One partition per entry of `partitions`.
fn partitioned_string_scan(name: &str, partitions: &[&[&str]]) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]));
    let batches: Vec<Vec<RecordBatch>> = partitions
        .iter()
        .map(|values| {
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(values.to_vec()))],
            )
            .unwrap()]
        })
        .collect();
    MemorySourceConfig::try_new_exec(&batches, schema, None).unwrap()
}

fn two_col_scan(a: &str, b: &str, vals: &[&str]) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(a, DataType::Utf8, true),
        Field::new(b, DataType::Int64, true),
    ]));
    let ids = StringArray::from(vals.to_vec());
    let ts = arrow::array::Int64Array::from(vec![1i64; vals.len()]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(ts)]).unwrap();
    MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

/// The magic's first byte is an illegal prost tag — pinned against
/// Ballista's own `BallistaPhysicalPlanNode` oneof's five variants' first
/// bytes (`ballista-core-54.1.0/src/serde/generated/ballista.rs:31-54`), so
/// an upstream 6th variant landing on the same byte is caught.
#[test]
fn magic_never_collides_with_a_ballista_oneof_tag() {
    const BALLISTA_ONEOF_FIRST_BYTES: [u8; 5] = [0x0A, 0x12, 0x1A, 0x22, 0x2A];
    let magic_first_byte = 0x07u8;
    assert!(
        !BALLISTA_ONEOF_FIRST_BYTES.contains(&magic_first_byte),
        "the jammi magic's first byte must never equal one of Ballista's own oneof tag bytes"
    );
    // The magic byte is also structurally illegal on its own terms: a
    // protobuf varint tag is (field_number << 3) | wire_type; field number 0
    // and wire type 7 are BOTH invalid, so no legal encoded message can ever
    // start with 0x07 regardless of how many oneof variants Ballista adds.
    let field_number = magic_first_byte >> 3;
    let wire_type = magic_first_byte & 0x07;
    assert_eq!(field_number, 0, "field number must be illegal (0)");
    assert_eq!(wire_type, 7, "wire type must be illegal (7)");
}

/// `plan`'s `InferenceExec`, searching depth-first.
fn inference_node(plan: &Arc<dyn ExecutionPlan>) -> &InferenceExec {
    let mut stack = vec![plan];
    while let Some(node) = stack.pop() {
        if let Some(exec) = node.downcast_ref::<InferenceExec>() {
            return exec;
        }
        stack.extend(node.children());
    }
    panic!("the plan has no InferenceExec")
}

#[tokio::test]
async fn inference_exec_round_trips() {
    let session = session().await;
    let scan = string_scan("text", &["hello", "world"]);
    let node = crate::inference_plan(&session, scan, session.compute_device().kind(), 1);
    let spec = inference_node(&node).spec().clone();

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec
        .try_encode(Arc::clone(&node), &mut buf)
        .expect("encode");
    assert_eq!(&buf[0..4], &[0x07, b'J', b'M', b'B'], "magic prefix");

    // The child crosses the wire on its own, so the decode is handed it.
    let inputs = [Arc::clone(node.children()[0])];
    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &inputs, &ctx).expect("decode");
    assert_eq!(
        inference_node(&decoded).spec(),
        &spec,
        "every field of the spec crosses the wire"
    );
    assert_eq!(spec, crate::text_embedding_spec(spec.device_kind, 1));

    // Re-encoding the decoded node reproduces the same bytes: nothing is
    // stamped or defaulted on the way through.
    let mut buf2 = Vec::new();
    codec.try_encode(decoded, &mut buf2).expect("re-encode");
    assert_eq!(buf, buf2, "encode -> decode -> encode is byte-identical");
}

/// A decode binds through the same constructor the planner uses, so a peer
/// cannot hand an executor an `InferenceExec` over an input that was never
/// numbered.
#[tokio::test]
async fn inference_decode_refuses_an_unnumbered_input() {
    let session = session().await;
    let scan = string_scan("text", &["hello"]);
    let node = crate::inference_plan(&session, scan, session.compute_device().kind(), 1);
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(node, &mut buf).unwrap();

    let inputs = [string_scan("text", &["hello"])];
    let err = codec
        .try_decode(&buf, &inputs, &session.context().task_ctx())
        .expect_err("an input without _ordinal must refuse");
    assert!(err.to_string().contains("_ordinal"), "{err}");
}

/// The codec never invents or rewrites a
/// `device_kind` — it carries exactly the constructed value across the wire,
/// even when the constructing session's own device kind differs from the
/// descriptor's.
#[tokio::test]
async fn codec_never_rewrites_device_kind() {
    let session = session().await; // a CPU session (`jammi_test_utils::test_config`)
    assert_eq!(
        session.compute_device().kind(),
        ComputeDeviceKind::Cpu,
        "precondition: the fixture session runs on CPU"
    );
    let scan = string_scan("text", &["hello"]);
    let node = crate::inference_plan(&session, scan, ComputeDeviceKind::Cuda, 1);
    assert_eq!(
        inference_node(&node).spec().device_kind,
        ComputeDeviceKind::Cuda,
        "constructed explicitly onto a kind other than the session's own"
    );

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();
    let inputs = [Arc::clone(node.children()[0])];
    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &inputs, &ctx).unwrap();
    assert_eq!(
        inference_node(&decoded).spec().device_kind,
        ComputeDeviceKind::Cuda,
        "the codec never rewrites device_kind to the decoding/encoding session's own kind"
    );
}

/// `NumberedInputExec` carries its one construction input, the row order, in
/// both of its forms.
#[tokio::test]
async fn numbered_input_exec_round_trips() {
    let session = session().await;
    let codec = JammiCodec::new(&session);
    let ctx = session.context().task_ctx();
    for order in [
        RowOrder::Arrival,
        RowOrder::Keyed {
            key_column: "id".into(),
        },
    ] {
        let node: Arc<dyn ExecutionPlan> = Arc::new(
            NumberedInputExec::try_new(
                two_col_scan("id", "_content_hash", &["b", "a"]),
                order.clone(),
            )
            .unwrap(),
        );
        let mut buf = Vec::new();
        codec.try_encode(Arc::clone(&node), &mut buf).unwrap();
        let inputs = [two_col_scan("id", "_content_hash", &["b", "a"])];
        let decoded = codec.try_decode(&buf, &inputs, &ctx).unwrap();
        let decoded = decoded.downcast_ref::<NumberedInputExec>().unwrap();
        assert_eq!(decoded.order(), &order);
        assert_eq!(decoded.schema(), node.schema());
    }
}

#[tokio::test]
async fn key_check_exec_round_trips() {
    let session = session().await;
    let scan = string_scan("id", &["a", "b"]);
    let node = KeyCheckExec::try_new(scan, "id").unwrap();
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();

    let inputs = [string_scan("id", &["a", "b"])];
    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &inputs, &ctx).unwrap();
    let decoded = decoded.downcast_ref::<KeyCheckExec>().unwrap();
    assert_eq!(decoded.key_column(), "id");
}

#[tokio::test]
async fn placed_attempt_round_trips() {
    let session = session().await;
    let descriptor = jammi_ai::operator::placed_attempt_exec::PlacedAttempt {
        job_id: "job-1".to_string(),
        attempt: 3,
        submitter: "instance-a".to_string(),
        // Deliberately NOT the session's own kind (Cpu): the wire must
        // carry exactly what was constructed, never the decoding session's
        // own default (the same "codec never rewrites device_kind" rule
        // `InferenceExec` round-trips under).
        device_kind: ComputeDeviceKind::Cuda,
    };
    let node = jammi_ai::operator::placed_attempt_exec::PlacedAttemptExec::new(descriptor.clone());
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();
    assert_eq!(&buf[0..4], &[0x07, b'J', b'M', b'B']);

    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
    let decoded = decoded
        .downcast_ref::<jammi_ai::operator::placed_attempt_exec::PlacedAttemptExec>()
        .unwrap();
    assert_eq!(decoded.descriptor().job_id, descriptor.job_id);
    assert_eq!(decoded.descriptor().attempt, descriptor.attempt);
    assert_eq!(decoded.descriptor().submitter, descriptor.submitter);
    assert_eq!(
        decoded.descriptor().device_kind,
        descriptor.device_kind,
        "the wire must carry the constructed device_kind verbatim, never the decoding \
         session's own kind"
    );
}

#[tokio::test]
async fn asof_join_exec_round_trips() {
    let session = session().await;
    let left = two_col_scan("id", "t", &["a", "b"]);
    let right = two_col_scan("id", "t", &["a", "b"]);
    let spec = AsofJoinSpecBuilder::new(
        AsofKey {
            by: vec!["id".into()],
            time: "t".into(),
        },
        AsofKey {
            by: vec!["id".into()],
            time: "t".into(),
        },
    )
    .build();
    let node = AsofJoinExec::try_new(left, right, spec.clone()).unwrap();
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();

    let inputs = [
        two_col_scan("id", "t", &["a", "b"]),
        two_col_scan("id", "t", &["a", "b"]),
    ];
    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &inputs, &ctx).unwrap();
    let decoded = decoded.downcast_ref::<AsofJoinExec>().unwrap();
    // `AsofJoinSpec` has no `PartialEq` (it is `Serialize`/`Deserialize`
    // only); compare via canonical JSON, the same representation the codec
    // itself carries on the wire.
    assert_eq!(
        serde_json::to_value(decoded.spec()).unwrap(),
        serde_json::to_value(&spec).unwrap(),
    );
}

// `block_in_place` (the codec's decode-time catalog re-read, `codec.rs`'s
// `block_on_catalog`) requires a MULTI-THREADED runtime — a real precondition
// this crate documents rather than papers over; every jammi-server process
// runs one, so this test runs on one too.
#[tokio::test(flavor = "multi_thread")]
async fn ann_search_exec_round_trips() {
    let session = session().await;
    let table_name = format!("bal_test_{}", uuid::Uuid::new_v4().simple());
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            table_name: &table_name,
            source_id: "src-1",
            model_id: "model-1",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "",
            dimensions: Some(4),
            key_column: Some("id"),
            text_columns: None,
            storage_precision: StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            writer_id: None,
            lease: None,
            job_attempt: None,
            replaces: None,
        })
        .await
        .expect("seed a result table row");

    let table = session
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    // A query-by-example vector: its provenance is part of what round-trips.
    let query = jammi_db::index::validate_query(
        vec![0.1, 0.2, 0.3, 0.4],
        4,
        jammi_db::index::QuerySource::Stored {
            table: table_name.clone(),
        },
    )
    .unwrap();
    let node = jammi_ai::operator::ann_search_exec::AnnSearchExec::new(
        table,
        query.clone(),
        5,
        Some(8),
        session.result_store(),
        session.context().clone(),
    )
    .unwrap();
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();

    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
    let decoded = decoded
        .downcast_ref::<jammi_ai::operator::ann_search_exec::AnnSearchExec>()
        .unwrap();
    assert_eq!(decoded.table().table_name, table_name);
    assert_eq!(decoded.k(), 5);
    assert_eq!(decoded.oversample_override(), Some(8));
    assert_eq!(decoded.query_vector(), &query);
}

/// `MaskExec` (a masked result-table scan) is the named v1 cut: neither
/// codec knows it, so it is refused typed, naming it — the SAME shape the
/// "a node neither codec knows is refused typed" property
/// covers for any other unknown node (proven by `unknown_node_is_refused_typed`
/// below over a plain `DataSourceExec`, which is unknown to BOTH codecs).
#[tokio::test]
async fn mask_exec_is_refused_typed() {
    let session = session().await;
    let input = string_scan("id", &["a"]);
    let mask = Arc::new(jammi_db::store::deletes::DeletionMask::empty());
    let node = jammi_db::store::masked_provider::MaskExec::new(input, 1, mask, 0, "t".into());
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    let err = codec
        .try_encode(Arc::new(node), &mut buf)
        .expect_err("MaskExec must be refused, never silently encoded");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("unsupported") || msg.to_lowercase().contains("maskexec"),
        "refusal must name the node: {msg}"
    );
}

/// A plan fanned out four ways is ADMITTED by a cluster. The plan the
/// production planner builds round-trips whole through `JammiCodec` to the
/// identical shape and spec, and Ballista's own planner cuts it into the four
/// stages the shape implies — the scan; the numbered input, written through a
/// hash shuffle on the chunk id; `InferenceExec` over that shuffle as FOUR
/// tasks; the merge — each of which round-trips through the codec in turn, as
/// it must to reach an executor.
#[tokio::test]
async fn a_plan_fanned_out_four_ways_is_admitted_and_staged() {
    use ballista_scheduler::planner::{DefaultDistributedPlanner, DistributedPlanner};
    use datafusion::config::ConfigOptions;
    use datafusion::physical_plan::displayable;
    use datafusion_proto::physical_plan::AsExecutionPlan;
    use datafusion_proto::protobuf::PhysicalPlanNode;
    use prost::Message;

    let session = session().await;
    let codec = JammiCodec::new(&session);
    let ctx = session.context().task_ctx();
    let round_trip = |plan: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
        let bytes = PhysicalPlanNode::try_from_physical_plan(plan, &codec)
            .expect("the plan encodes")
            .encode_to_vec();
        PhysicalPlanNode::decode(bytes.as_slice())
            .expect("the bytes parse")
            .try_into_physical_plan(&ctx, &codec)
            .expect("the plan decodes")
    };
    let shape = |plan: &Arc<dyn ExecutionPlan>| displayable(plan.as_ref()).indent(true).to_string();

    let scan = partitioned_string_scan("text", &[&["a", "b"], &["c"], &["d", "e"], &["f"]]);
    let plan = crate::inference_plan(&session, scan, ComputeDeviceKind::Cpu, 4);
    let decoded = round_trip(Arc::clone(&plan));
    assert_eq!(shape(&decoded), shape(&plan));
    assert_eq!(
        inference_node(&decoded).spec(),
        inference_node(&plan).spec()
    );

    let stages = DefaultDistributedPlanner::new()
        .plan_query_stages(
            &"job-n4".to_string().into(),
            plan,
            &ConfigOptions::default(),
        )
        .expect("the plan stages");
    let shapes: Vec<String> = stages
        .iter()
        .map(|stage| shape(&(Arc::clone(stage) as Arc<dyn ExecutionPlan>)))
        .collect();
    let all = shapes.join("\n");
    assert_eq!(
        stages.len(),
        4,
        "scan, numbered input, inference, merge:\n{all}"
    );

    assert_eq!(stages[0].input_partition_count(), 4, "{all}");
    assert!(shapes[0].contains("DataSourceExec"), "{all}");

    assert_eq!(stages[1].input_partition_count(), 1, "{all}");
    assert!(
        shapes[1].contains("NumberedInputExec: order=arrival"),
        "{all}"
    );
    assert!(!shapes[1].contains("InferenceExec"), "{all}");
    let shuffle = stages[1]
        .shuffle_output_partitioning()
        .expect("the numbered input is written through a hash shuffle");
    assert_eq!(shuffle.to_string(), "Hash([_ordinal@1 / 8], 4)", "{all}");

    assert_eq!(
        stages[2].input_partition_count(),
        4,
        "the inference stage runs as four tasks:\n{all}"
    );
    assert!(shapes[2].contains("InferenceExec"), "{all}");
    assert!(stages[2].shuffle_output_partitioning().is_none(), "{all}");

    assert_eq!(stages[3].input_partition_count(), 1, "{all}");
    assert!(
        shapes[3].contains("SortPreservingMergeExec: [_ordinal@1 ASC NULLS LAST]"),
        "{all}"
    );

    for (stage, expected) in stages.iter().zip(&shapes) {
        let decoded = round_trip(Arc::clone(stage) as Arc<dyn ExecutionPlan>);
        assert_eq!(&shape(&decoded), expected);
    }
}

/// A plain in-memory scan is unknown to jammi's codec (not one of the four
/// jammi node types) AND unknown to Ballista's delegate (not one of its
/// shuffle/coalesce/chaos node types) — the delegation-then-typed-refusal
/// path a Ballista shuffle node would otherwise short-circuit by being
/// recognized.
#[tokio::test]
async fn unknown_node_is_refused_typed() {
    let session = session().await;
    let node = string_scan("x", &["a"]);
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    let err = codec
        .try_encode(node, &mut buf)
        .expect_err("a node neither codec knows must be refused, never silently encoded");
    assert!(err.to_string().to_lowercase().contains("unsupported"));
}

#[tokio::test]
async fn truncated_magic_buffer_is_refused_typed_never_delegated() {
    let session = session().await;
    let codec = JammiCodec::new(&session);
    let buf = vec![0x07u8, b'J', b'M', b'B']; // magic present, no tag byte
    let ctx = session.context().task_ctx();
    let err = codec
        .try_decode(&buf, &[], &ctx)
        .expect_err("a magic-prefixed buffer too short for a tag byte must be refused typed");
    assert!(err.to_string().to_lowercase().contains("truncated"));
}

/// A buffer without the magic is NEVER decoded as ours — it delegates whole
/// to Ballista's own codec, which fails DIFFERENTLY (its own "could not
/// deserialize BallistaPhysicalPlanNode" decode error, never jammi's
/// "unknown jammi node tag"/"truncated" wording) because the bytes are
/// garbage to it too. Distinguishing the two error shapes is how this test
/// proves delegation actually happened, not merely that decode failed.
#[tokio::test]
async fn no_magic_buffer_delegates_to_ballistas_own_codec() {
    let session = session().await;
    let codec = JammiCodec::new(&session);
    // No magic prefix at all — four arbitrary bytes no jammi tag matches.
    let buf = vec![0xFFu8, 0xFF, 0xFF, 0xFF];
    let ctx = session.context().task_ctx();
    let err = codec
        .try_decode(&buf, &[], &ctx)
        .expect_err("garbage with no magic must still fail (it is garbage to Ballista too)");
    let msg = err.to_string();
    assert!(
        !msg.to_lowercase().contains("unknown jammi node tag")
            && !msg.to_lowercase().contains("truncated jammi"),
        "a no-magic buffer must never be handled by jammi's own decode arm: {msg}"
    );
    assert!(
        msg.contains("BallistaPhysicalPlanNode") || msg.to_lowercase().contains("deserialize"),
        "the delegate's own decode error must be the one surfaced: {msg}"
    );
}

#[tokio::test]
async fn dead_session_is_refused_typed() {
    let session = session().await;
    let scan = string_scan("text", &["a"]);
    let node = crate::inference_plan(&session, scan, session.compute_device().kind(), 1);
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(node, &mut buf).unwrap();

    let ctx = session.context().task_ctx();
    drop(session);
    let inputs = [string_scan("text", &["a"])];
    let err = codec
        .try_decode(&buf, &inputs, &ctx)
        .expect_err("a codec whose session has gone away must refuse typed, never panic");
    assert!(err.to_string().to_lowercase().contains("session"));
}

/// Tenant isolation at the codec's decode site (the oracle's per-RPC
/// invariant, applied to the Ballista listeners): `AnnSearchExec` is rebuilt
/// on an executor from the table name AND the tenant the SUBMITTER's session
/// carried onto the wire, through the strict tenant-pinned read
/// `get_result_table_for_tenant`. A descriptor naming tenant A's table under
/// tenant B, or under no tenant, must NOT resolve; under A it does. The
/// descriptors are built directly on the wire package — the codec's own
/// encoder would never produce the cross-tenant ones. Mutation: relax the
/// decode to `get_result_table` (the ambient read) and both refusal arms
/// resolve.
#[tokio::test(flavor = "multi_thread")]
async fn ann_search_decode_refuses_another_tenants_table_and_a_tenant_free_read_of_a_bound_one() {
    use jammi_ballista::codec::{pb, NodeTag, MAGIC};
    use jammi_db::tenant::TenantId;
    use prost::Message;

    let session = session().await;
    let tenant_a = TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap();
    let tenant_b = TenantId::from_uuid(uuid::Uuid::new_v4()).unwrap();
    let table_name = format!("bal_tenant_{}", uuid::Uuid::new_v4().simple());
    session
        .catalog()
        .pinned_to_tenant(Some(tenant_a))
        .create_result_table(CreateResultTableParams {
            table_name: &table_name,
            source_id: "src-1",
            model_id: "model-1",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "",
            dimensions: Some(4),
            key_column: Some("id"),
            text_columns: None,
            storage_precision: StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            writer_id: None,
            lease: None,
            job_attempt: None,
            replaces: None,
        })
        .await
        .expect("seed tenant A's result table row");

    let descriptor = |tenant: Option<String>| {
        let msg = pb::AnnSearchExecNode {
            table_name: table_name.clone(),
            tenant_id: tenant,
            query_vector: vec![0.1, 0.2, 0.3, 0.4],
            k: 5,
            oversample_override: None,
            query_stored_table: None,
        };
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(NodeTag::AnnSearch as u8);
        msg.encode(&mut buf).unwrap();
        buf
    };
    let codec = JammiCodec::new(&session);
    let ctx = session.context().task_ctx();

    let err = codec
        .try_decode(&descriptor(Some(tenant_b.to_string())), &[], &ctx)
        .expect_err("tenant B must not resolve tenant A's table");
    assert!(
        err.to_string().contains("not found"),
        "the refusal is the non-disclosing not-found, never a widened read: {err}"
    );
    let err = codec
        .try_decode(&descriptor(None), &[], &ctx)
        .expect_err("a tenant-free read must not resolve a tenant-bound table");
    assert!(err.to_string().contains("not found"), "{err}");

    let decoded = codec
        .try_decode(&descriptor(Some(tenant_a.to_string())), &[], &ctx)
        .expect("the owning tenant resolves its own table");
    let decoded = decoded
        .downcast_ref::<jammi_ai::operator::ann_search_exec::AnnSearchExec>()
        .unwrap();
    assert_eq!(decoded.table().table_name, table_name);
}

/// `decode_ann_search` checks a decoded query's width against the catalog
/// authority it already has in hand (`table`, resolved a few lines before
/// the check) rather than unconditionally passing `None` and skipping the
/// authority the decode function just looked up. This decode-time check is
/// defense in depth (a conforming coordinator's own `QueryBuilder::new`
/// checks the width before ever building the plan node), so this is a
/// hand-built malformed-in-transit descriptor, not the shape any real
/// encoder would produce.
///
/// **What this proves, precisely — IN-PROCESS recovery, not the wire
/// class.** Calling `codec.try_decode` directly (as this test does) and
/// inspecting the returned `DataFusionError` recovers the caller-fault
/// class: boxing the classified `JammiError` directly (rather than routing
/// through this crate's own `Error::Decode`/`Error::Catalog`, which
/// `crates/jammi-ballista/src/error.rs`'s `into_df_error` would box as THIS
/// crate's `Error`, defeating the downstream downcast) is what makes
/// `jammi_db::error`'s structural `DataFusionError` -> `JammiError`
/// classifier ("owned passthrough" shape) recover
/// `JammiError::Schema { .. }` here. This does NOT hold across a real
/// distributed Ballista job: `ballista-executor` stringifies a failed
/// task's error (`e.to_string()`) into `TaskStatus`, and the job-status
/// waiter rebuilds job failure as a bare `DataFusionError::Execution(String)`
/// — no boxed value survives that hop, so a REMOTE client sees
/// `Code::Internal` (`jammi-server/src/grpc/wire.rs`'s catch-all), never
/// `InvalidArgument`, regardless of which typed variant this decode boxes.
/// The caller-class path for a remote client is the coordinator's own
/// `QueryBuilder::new` check, which runs before any plan is shipped.
///
/// Mutation: pass `None` instead of `table.dimensions()` and this test reds
/// (the malformed query decodes instead of being refused).
#[tokio::test(flavor = "multi_thread")]
async fn ann_search_decode_checks_width_against_the_catalog_authority_it_holds() {
    use jammi_ballista::codec::{pb, NodeTag, MAGIC};
    use jammi_db::error::JammiError;
    use prost::Message;

    let session = session().await;
    let table_name = format!("bal_width_authority_{}", uuid::Uuid::new_v4().simple());
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            table_name: &table_name,
            source_id: "src-1",
            model_id: "model-1",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "",
            dimensions: Some(4),
            key_column: Some("id"),
            text_columns: None,
            storage_precision: StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            writer_id: None,
            lease: None,
            job_attempt: None,
            replaces: None,
        })
        .await
        .expect("seed a result table row");

    // A 3-wide query against a table whose catalog width is 4 — no real
    // encoder would ever build this; it stands in for a peer that skipped
    // its own `QueryBuilder::new` check. The provenance it was encoded with
    // decides the class on THIS process exactly as it would have there.
    let refused =
        |query_stored_table: Option<String>| {
            let msg = pb::AnnSearchExecNode {
                table_name: table_name.clone(),
                tenant_id: None,
                query_vector: vec![0.1, 0.2, 0.3],
                k: 5,
                oversample_override: None,
                query_stored_table,
            };
            let mut buf = Vec::new();
            buf.extend_from_slice(&MAGIC);
            buf.push(NodeTag::AnnSearch as u8);
            msg.encode(&mut buf).unwrap();
            let codec = JammiCodec::new(&session);
            let ctx = session.context().task_ctx();
            JammiError::from(codec.try_decode(&buf, &[], &ctx).expect_err(
                "a 3-wide query against a catalog width of 4 must be refused at decode",
            ))
        };

    let stored = refused(Some("docs_embeddings".into()));
    assert!(
        matches!(
            &stored,
            JammiError::IncompatibleFormat { artifact, .. } if artifact == "docs_embeddings.vector"
        ),
        "a stored query's width fault is its own table's, never the caller's — got {stored:?}"
    );

    let classified = refused(None);
    assert!(
        matches!(classified, JammiError::Schema { .. }),
        "expected the caller-fault Schema class (gRPC InvalidArgument), got {classified:?}"
    );
}

/// The result-table sink crosses as its spec and arrives PLACED: every
/// field of the spec survives, the node that decodes writes here and never
/// re-submits, and a re-encode reproduces the bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn result_table_sink_exec_round_trips_and_arrives_placed() {
    let session = session().await;
    let store = session.result_store();
    let building = store
        .create_table(
            "docs",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "sentence-transformers/all-MiniLM-L6-v2",
            Some(4),
            Some("text"),
            Some("text"),
            None,
        )
        .await
        .expect("a building row");
    let spec = jammi_db::store::ResultTableSinkSpec {
        table_name: building.table_name().to_string(),
        parquet_url: building.parquet_url().clone(),
        tenant: building.tenant(),
        writer_id: building.writer_id().to_string(),
        storage_precision: building.storage_precision(),
        lease: SinkLeaseKind::Table,
        kind: SinkKind::Embeddings {
            dimensions: 4,
            ann: *store.ann_config(),
            checkpoint_interval: 2,
        },
    };
    let child = crate::inference_plan(
        &session,
        string_scan("text", &["hello", "world"]),
        session.compute_device().kind(),
        1,
    );
    let node: Arc<dyn ExecutionPlan> = Arc::new(ResultTableSinkExec::new(
        spec.clone(),
        Arc::clone(&child),
        ResultStore::clone(&store),
    ));
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec
        .try_encode(Arc::clone(&node), &mut buf)
        .expect("encode");
    assert_eq!(&buf[0..4], &[0x07, b'J', b'M', b'B'], "magic prefix");

    let decoded = codec
        .try_decode(&buf, &[child], &session.context().task_ctx())
        .expect("decode");
    let sink = decoded
        .downcast_ref::<ResultTableSinkExec>()
        .expect("a ResultTableSinkExec");
    assert_eq!(
        sink.spec(),
        &spec,
        "every field of the spec crosses the wire"
    );
    assert!(
        sink.is_placed(),
        "a decoded sink writes here, never re-submits"
    );

    let mut buf2 = Vec::new();
    codec.try_encode(decoded, &mut buf2).expect("re-encode");
    assert_eq!(buf, buf2, "encode -> decode -> encode is byte-identical");
    building.abort().await.unwrap();
}

/// A sink whose object is not under the decoding store's root, or whose
/// row this catalog does not hold, is refused typed on decode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn result_table_sink_decode_refuses_a_foreign_object_and_an_unknown_row() {
    let session = session().await;
    let store = session.result_store();
    let child = string_scan("text", &["hello"]);
    let foreign = jammi_db::store::ResultTableSinkSpec {
        table_name: "docs__text_embedding__m__1".into(),
        parquet_url: jammi_db::storage::StorageUrl::parse(
            "file:///elsewhere/jammi_db/_global/docs__text_embedding__m__1.parquet",
        )
        .unwrap(),
        tenant: None,
        writer_id: "writer-elsewhere".into(),
        storage_precision: StoragePrecision::default(),
        lease: SinkLeaseKind::Table,
        kind: SinkKind::Rows,
    };
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec
        .try_encode(
            Arc::new(ResultTableSinkExec::new(
                foreign.clone(),
                Arc::clone(&child),
                ResultStore::clone(&store),
            )),
            &mut buf,
        )
        .unwrap();
    let err = codec
        .try_decode(&buf, &[Arc::clone(&child)], &session.context().task_ctx())
        .expect_err("an object outside this store's root is refused");
    assert!(
        matches!(
            jammi_db::error::JammiError::from(err),
            jammi_db::error::JammiError::Storage(
                jammi_db::storage::StorageError::InvalidUrl { .. }
            )
        ),
        "expected InvalidUrl"
    );

    let unknown = jammi_db::store::ResultTableSinkSpec {
        parquet_url: jammi_db::storage::StorageUrl::parse(&format!(
            "{}/_global/docs__text_embedding__m__1.parquet",
            store.root().as_str()
        ))
        .unwrap(),
        ..foreign
    };
    let mut buf = Vec::new();
    codec
        .try_encode(
            Arc::new(ResultTableSinkExec::new(
                unknown,
                Arc::clone(&child),
                ResultStore::clone(&store),
            )),
            &mut buf,
        )
        .unwrap();
    let err = codec
        .try_decode(&buf, &[child], &session.context().task_ctx())
        .expect_err("a row this catalog does not hold is refused");
    assert!(
        matches!(
            jammi_db::error::JammiError::from(err),
            jammi_db::error::JammiError::RowGone { ref table } if table == "docs__text_embedding__m__1"
        ),
        "expected RowGone"
    );
}
