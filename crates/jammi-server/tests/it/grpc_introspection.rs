//! Engine introspection over the wire, proven interchangeable with the local
//! transport: `list_sources` / `describe_source` and `server_info`
//! (`CatalogService.ListSources` / `DescribeSource` / `GetServerInfo`).
//!
//! An in-process gRPC chain hosts a real `InferenceSession`; a
//! `jammi_admin::CatalogClient` connects over a real HTTP/2 channel and a
//! `jammi_ai::Session` wraps the *same* engine `Arc`, so any divergence is the
//! transport's fault, not the engine's. The three properties this pins:
//!
//! * **Descriptor parity** — after a real source is registered and embedded
//!   (realistic `tiny_bert` text embeddings over the bundled `patents` corpus,
//!   never dummy vectors), `list_sources` / `describe_source` return the same
//!   descriptor through either transport: same registry identity and the same
//!   embedding `status` / `row_count` / `dimensions` read off the result table,
//!   the source-of-truth a `generate_embeddings` response also returns.
//! * **Absent-source parity** — `describe_source` of an unregistered id returns
//!   `None` on both transports (the remote arm maps the server's `NotFound`
//!   back to `None`), never a faked empty descriptor.
//! * **Server-info parity** — `server_info` returns the same version, features,
//!   and storage backends through either transport; both always carry `file`
//!   and `memory` backends, sorted and de-duplicated.
//!
//! Hermetic: the encoder is the local `tiny_bert` cookbook fixture and the
//! corpus is the bundled `patents.parquet`; no live network, no download.

use std::sync::Arc;

use jammi_admin::CatalogClient;
use jammi_ai::{Modality, ServerInfo, Session, SourceDescriptor};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_test_utils::{cookbook_fixture, fixture};
use tonic::transport::Endpoint;

use super::common::grpc::{start_engine_server, EngineServer};

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn patents_connection() -> SourceConnection {
    SourceConnection {
        url: Some(format!("file://{}", fixture("patents.parquet").display())),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }
}

async fn remote(server: &EngineServer) -> CatalogClient {
    let endpoint = Endpoint::from_shared(format!("http://{}", server.addr)).expect("endpoint");
    CatalogClient::connect(endpoint)
        .await
        .expect("catalog client connect")
}

fn local(server: &EngineServer) -> Session {
    Session::new(Arc::clone(&server.engine))
}

/// A comparable projection of a descriptor: the registry identity plus, per
/// result table, the client-observable embedding fields (`table_name`,
/// `status`, `row_count`, `dimensions`, `task`). Comparing on this rather than
/// the whole record keeps the assertion on the fields the verb surfaces while
/// ignoring server-internal bookkeeping the wire intentionally drops
/// (parquet/index paths, timestamps).
type TableShape = (String, String, usize, Option<i32>, jammi_db::ModelTask);
type DescriptorShape = (String, SourceType, String, Vec<TableShape>);

fn descriptor_shape(d: &SourceDescriptor) -> DescriptorShape {
    let mut tables: Vec<TableShape> = d
        .result_tables
        .iter()
        .map(|t| {
            (
                t.table_name.clone(),
                t.status.clone(),
                t.row_count,
                t.dimensions,
                t.task,
            )
        })
        .collect();
    tables.sort_by(|a, b| a.0.cmp(&b.0));
    (
        d.source_id.clone(),
        d.source_type.clone(),
        d.status.clone(),
        tables,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_list_and_describe_sources_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);

    // Register + embed the corpus on the shared engine through the local
    // session, so both transports introspect a real, embedded source.
    local
        .add_source("patents", SourceType::File, patents_connection())
        .await
        .expect("add_source");
    let table = local
        .generate_embeddings(
            "patents",
            &tiny_bert_model_id(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
        )
        .await
        .expect("generate_embeddings");
    assert_eq!(table.status, "ready");
    assert!(table.row_count > 0, "patents corpus embeds rows");
    assert!(table.dimensions.is_some(), "dimensions recorded");

    // list_sources parity: same descriptors through either transport.
    let remote_list = remote.list_sources().await.expect("remote list_sources");
    let local_list = local.list_sources().await.expect("local list_sources");
    assert_eq!(remote_list.len(), 1, "exactly the one registered source");
    let remote_shapes: Vec<_> = remote_list.iter().map(descriptor_shape).collect();
    let local_shapes: Vec<_> = local_list.iter().map(descriptor_shape).collect();
    assert_eq!(
        remote_shapes, local_shapes,
        "list_sources returns the same descriptors through either transport"
    );

    // The descriptor carries the embedding numbers from their source-of-truth:
    // the result table generate_embeddings just produced.
    let described = remote_list.into_iter().next().expect("one source");
    assert_eq!(described.source_id, "patents");
    assert_eq!(described.source_type, SourceType::File);
    assert_eq!(described.result_tables.len(), 1, "one embedding table");
    let rt = &described.result_tables[0];
    assert_eq!(rt.status, "ready");
    assert_eq!(rt.row_count, table.row_count);
    assert_eq!(rt.dimensions, table.dimensions);
    assert_eq!(rt.task, jammi_db::ModelTask::TextEmbedding);

    // describe_source parity for a present source.
    let remote_one = remote
        .describe_source("patents")
        .await
        .expect("remote describe_source")
        .expect("patents is registered");
    let local_one = local
        .describe_source("patents")
        .await
        .expect("local describe_source")
        .expect("patents is registered");
    assert_eq!(
        descriptor_shape(&remote_one),
        descriptor_shape(&local_one),
        "describe_source returns the same descriptor through either transport"
    );

    // Absent-source parity: both return None (the remote arm maps the server's
    // NotFound back to None), never a faked descriptor.
    let remote_absent = remote
        .describe_source("no-such-source")
        .await
        .expect("remote describe_source of an absent id is not an error");
    let local_absent = local
        .describe_source("no-such-source")
        .await
        .expect("local describe_source of an absent id is not an error");
    assert!(remote_absent.is_none(), "remote: absent source → None");
    assert!(local_absent.is_none(), "local: absent source → None");

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_server_info_like_local() {
    let server = start_engine_server().await;
    let remote = remote(&server).await;
    let local = local(&server);

    let remote_info = remote.server_info().await.expect("remote server_info");
    let local_info = local.server_info().await.expect("local server_info");

    // The compile-time facts (version / features / storage_backends) are the
    // same through either transport — they describe the build, not the
    // deployment.
    assert_eq!(
        (
            &remote_info.version,
            &remote_info.features,
            &remote_info.storage_backends
        ),
        (
            &local_info.version,
            &local_info.features,
            &local_info.storage_backends
        ),
        "the compile-time self-description is identical across transports"
    );
    assert!(!local_info.version.is_empty(), "version is reported");
    // file + memory are always-present backends, regardless of feature set.
    assert!(local_info.storage_backends.contains(&"file".to_string()));
    assert!(local_info.storage_backends.contains(&"memory".to_string()));
    // The lists are sorted and de-duplicated.
    let mut sorted = local_info.storage_backends.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(local_info.storage_backends, sorted);

    // `services` is the runtime tier handshake and legitimately differs by
    // transport: the in-process `Session` mounts no gRPC services, so it
    // reports an empty set and equals the engine's compile-time self-description
    // (which leaves `services` empty); the remote server reports the tiers it
    // actually mounted. `start_engine_server` mounts the engine core + the
    // engine-backed optional tiers (eval, and train when compiled) but no
    // trigger, so `event` is absent while `core`/`eval` are present.
    assert_eq!(
        local_info,
        ServerInfo::current(),
        "the local session's self-description carries no mounted services"
    );
    assert!(
        local_info.services.is_empty(),
        "in-process: no gRPC services"
    );
    assert!(
        remote_info.services.contains(&"core".to_string()),
        "remote always mounts the core tier"
    );
    assert!(
        remote_info.services.contains(&"eval".to_string()),
        "this engine fixture mounts the eval tier"
    );
    assert!(
        !remote_info.services.contains(&"event".to_string()),
        "this fixture mounts no trigger, so the event tier is absent"
    );

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
