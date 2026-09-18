//! Hermetic codec round-trip oracles (contract `feat_500-wave4` §7 (a1)).

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;

use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::operator::inference_exec::InferenceExecBuilder;
use jammi_ai::operator::key_check_exec::KeyCheckExec;
use jammi_ai::pipeline::asof::exec::AsofJoinExec;
use jammi_ai::pipeline::asof::spec::{AsofJoinSpecBuilder, AsofKey};
use jammi_ai::session::InferenceSession;
use jammi_ballista::codec::JammiCodec;
use jammi_db::catalog::result_repo::CreateResultTableParams;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::config::StoragePrecision;
use jammi_db::store::manifest::ComputeDeviceKind;

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

/// A1: the magic's first byte is an illegal prost tag — pinned against
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

#[tokio::test]
async fn inference_exec_round_trips() {
    let session = session().await;
    let scan = string_scan("text", &["hello", "world"]);
    let node = InferenceExecBuilder::new(
        scan,
        ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2"),
        ModelTask::TextEmbedding,
        vec!["text".to_string()],
        "text".to_string(),
        "src-1".to_string(),
        Arc::clone(session.model_cache()),
        session.compute_device().kind(),
    )
    .batch_size(8)
    .embedding_dim(Some(4))
    .passthrough(vec![])
    .build()
    .expect("inference exec builds");
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec
        .try_encode(Arc::clone(&node), &mut buf)
        .expect("encode");
    assert_eq!(&buf[0..4], &[0x07, b'J', b'M', b'B'], "magic prefix");

    let inputs = [string_scan("text", &["hello", "world"])];
    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &inputs, &ctx).expect("decode");
    {
        let d = decoded
            .downcast_ref::<jammi_ai::operator::inference_exec::InferenceExec>()
            .expect("decodes to InferenceExec");
        assert_eq!(
            d.source(),
            &ModelSource::hf("sentence-transformers/all-MiniLM-L6-v2")
        );
        assert_eq!(d.task(), ModelTask::TextEmbedding);
        assert_eq!(d.content_columns(), &["text".to_string()]);
        assert_eq!(d.key_column(), "text");
        assert_eq!(d.source_id(), "src-1");
        assert_eq!(d.batch_size(), 8);
        assert_eq!(d.embedding_dim(), Some(4));
        assert_eq!(d.device_kind(), session.compute_device().kind());
    }

    // Re-encoding the SAME decoded `Arc<dyn ExecutionPlan>` reproduces the
    // same bytes: device_kind is stamped once, at construction, so the
    // second encode stamps nothing new.
    let mut buf2 = Vec::new();
    codec.try_encode(decoded, &mut buf2).expect("re-encode");
    assert_eq!(buf, buf2, "encode -> decode -> encode is byte-identical");
}

/// Replaces `inference_exec_device_kind_defaults_to_the_submitting_sessions_own`
/// (contract `feat_500-wave4` §9 B3): the codec never invents or rewrites a
/// `device_kind` — it carries exactly the constructed value across the wire,
/// even when the constructing session's own device kind differs from the
/// descriptor's.
#[tokio::test]
async fn codec_never_rewrites_device_kind() {
    let session = session().await; // a CPU session (`jammi_test_utils::test_config`)
    assert_eq!(
        session.compute_device().kind(),
        jammi_db::store::manifest::ComputeDeviceKind::Cpu,
        "precondition: the fixture session runs on CPU"
    );
    let scan = string_scan("text", &["hello"]);
    let node = InferenceExecBuilder::new(
        scan,
        ModelSource::hf("m"),
        ModelTask::TextEmbedding,
        vec!["text".to_string()],
        "text".to_string(),
        "src-1".to_string(),
        Arc::clone(session.model_cache()),
        jammi_db::store::manifest::ComputeDeviceKind::Cuda,
    )
    .embedding_dim(Some(2))
    .build()
    .unwrap();
    assert_eq!(
        node.device_kind(),
        jammi_db::store::manifest::ComputeDeviceKind::Cuda,
        "constructed explicitly onto a kind other than the session's own"
    );
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();
    let inputs = [string_scan("text", &["hello"])];
    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &inputs, &ctx).unwrap();
    let decoded = decoded
        .downcast_ref::<jammi_ai::operator::inference_exec::InferenceExec>()
        .unwrap();
    assert_eq!(
        decoded.device_kind(),
        jammi_db::store::manifest::ComputeDeviceKind::Cuda,
        "the codec never rewrites device_kind to the decoding/encoding session's own kind"
    );
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
async fn gang_exec_round_trips() {
    let session = session().await;
    let descriptor = jammi_ai::operator::gang_exec::GangDescriptor {
        job_id: "job-1".to_string(),
        attempt: 3,
        world: 2,
        submitter: "instance-a".to_string(),
        // Deliberately NOT the session's own kind (Cpu): the wire must
        // carry exactly what was constructed, never the decoding session's
        // own default (LANE pressure-round correction, the same "codec
        // never rewrites device_kind" rule `InferenceExec` round-trips
        // under).
        device_kind: ComputeDeviceKind::Cuda,
    };
    let node = jammi_ai::operator::gang_exec::GangExec::new(descriptor.clone());
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);

    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    codec.try_encode(Arc::clone(&node), &mut buf).unwrap();
    assert_eq!(&buf[0..4], &[0x07, b'J', b'M', b'B']);

    let ctx = session.context().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &ctx).unwrap();
    let decoded = decoded
        .downcast_ref::<jammi_ai::operator::gang_exec::GangExec>()
        .unwrap();
    assert_eq!(decoded.descriptor().job_id, descriptor.job_id);
    assert_eq!(decoded.descriptor().attempt, descriptor.attempt);
    assert_eq!(decoded.descriptor().world, descriptor.world);
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
// runs one, so this is the honest test shape, not a workaround.
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
        })
        .await
        .expect("seed a result table row");

    let table = session
        .catalog()
        .get_result_table(&table_name)
        .await
        .unwrap()
        .unwrap();
    let query = jammi_db::index::validate_query(
        vec![0.1, 0.2, 0.3, 0.4],
        None,
        jammi_db::index::QuerySource::Caller,
    )
    .unwrap();
    let node = jammi_ai::operator::ann_search_exec::AnnSearchExec::new(
        table,
        query,
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
    assert_eq!(decoded.query_vector().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
}

/// `MaskExec` (a masked result-table scan) is the named v1 cut: neither
/// codec knows it, so it is refused typed, naming it — the SAME shape the
/// contract's "a node neither codec knows is refused typed" property
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

/// RS6 (#540 RANGESPLIT), the contract's stop-rule exit: `OrdinalSplitExec`
/// gets the SAME v1-cut treatment as `MaskExec` above — no `NodeTag`, no
/// `plan.proto` message, no encode/decode arm — so a plan containing it
/// (only ever built when `InferenceConfig::partitions > 1`; the default `1`
/// never inserts this node) is refused typed rather than silently
/// mis-encoded or handed to Ballista's own delegate (which does not know it
/// either). See `jammi_ballista::codec`'s module doc for WHY: in Ballista
/// 54.1 a `SortPreservingMergeExec` is a stage boundary, so the N-partition
/// `InferenceExec(OrdinalSplitExec)` this node feeds would become its own
/// stage of N tasks each executing ONE partition, in general in a separate
/// process — this node's in-process shared-mutex mechanism has no meaning
/// across that split.
#[tokio::test]
async fn ordinal_split_exec_is_refused_typed() {
    let input = string_scan("id", &["a"]);
    let node = jammi_ai::operator::ordinal_split_exec::OrdinalSplitExec::new(input, 2)
        .expect("OrdinalSplitExec builds over a plain scan");
    let session = session().await;
    let codec = JammiCodec::new(&session);
    let mut buf = Vec::new();
    let err = codec
        .try_encode(Arc::new(node), &mut buf)
        .expect_err("OrdinalSplitExec must be refused, never silently encoded");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("unsupported")
            || msg.to_lowercase().contains("ordinalsplitexec"),
        "refusal must name the node: {msg}"
    );
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
    let node = InferenceExecBuilder::new(
        scan,
        ModelSource::hf("m"),
        ModelTask::TextEmbedding,
        vec!["text".to_string()],
        "text".to_string(),
        "src-1".to_string(),
        Arc::clone(session.model_cache()),
        session.compute_device().kind(),
    )
    .embedding_dim(Some(2))
    .build()
    .unwrap();
    let node: Arc<dyn ExecutionPlan> = Arc::new(node);
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
/// `QueryBuilder::new` check, which runs before any plan is shipped —
/// tracked as a residual on #519, its own unit, not a fold of this test.
///
/// Mutation executed: revert the fix (pass `None` instead of
/// `table.dimensions()`) — this test reds (the malformed query decodes
/// instead of being refused).
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
        })
        .await
        .expect("seed a result table row");

    // A 3-wide query against a table whose catalog width is 4 — no real
    // encoder would ever build this; it stands in for a peer that skipped
    // its own `QueryBuilder::new` check.
    let msg = pb::AnnSearchExecNode {
        table_name: table_name.clone(),
        tenant_id: None,
        query_vector: vec![0.1, 0.2, 0.3],
        k: 5,
        oversample_override: None,
    };
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::AnnSearch as u8);
    msg.encode(&mut buf).unwrap();

    let codec = JammiCodec::new(&session);
    let ctx = session.context().task_ctx();
    let err = codec
        .try_decode(&buf, &[], &ctx)
        .expect_err("a 3-wide query against a catalog width of 4 must be refused at decode");

    let classified = JammiError::from(err);
    assert!(
        matches!(classified, JammiError::Schema { .. }),
        "expected the caller-fault Schema class (gRPC InvalidArgument), got {classified:?}"
    );
}
