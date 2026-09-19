//! The object-store registry on a session's runtime environment is reachable
//! from every plan node; the stores it yields read, and refuse to mutate.

use datafusion::execution::object_store::ObjectStoreUrl;
use jammi_db::session::JammiSession;
use object_store::path::Path;
use object_store::{ObjectStoreExt, PutPayload};

use crate::common;

/// The `file://` store a session's registry resolves reads a model artifact
/// under the artifact root and refuses to delete or overwrite it.
#[tokio::test]
async fn the_sessions_registry_cannot_mutate_a_managed_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = JammiSession::new(common::test_config(dir.path()))
        .await
        .expect("session");

    let artifact = dir.path().join("models").join("adapter.safetensors");
    std::fs::create_dir_all(artifact.parent().expect("parent")).expect("models dir");
    std::fs::write(&artifact, b"weights").expect("seed the artifact");

    let store = session
        .context()
        .runtime_env()
        .object_store(ObjectStoreUrl::local_filesystem())
        .expect("the registry resolves file://");
    let key = Path::from_filesystem_path(&artifact).expect("an object path");

    let read = store
        .get(&key)
        .await
        .expect("reads pass")
        .bytes()
        .await
        .expect("bytes");
    assert_eq!(&read[..], b"weights");

    for refused in [
        store.delete(&key).await.err(),
        store
            .put(&key, PutPayload::from_static(b"overwritten"))
            .await
            .err(),
    ] {
        assert!(
            matches!(refused, Some(object_store::Error::NotSupported { .. })),
            "a mutation through the registry must be refused, got {refused:?}"
        );
    }
    assert_eq!(std::fs::read(&artifact).expect("still there"), b"weights");
}
