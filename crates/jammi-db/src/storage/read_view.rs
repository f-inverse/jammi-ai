//! The object stores DataFusion sees: read-only views.
//!
//! DataFusion resolves every scan through the object-store registry on its
//! runtime environment, and that registry is reachable from anywhere a
//! `SessionContext`, a `SessionState` or a `TaskContext` is — which includes
//! every `ExecutionPlan::execute` in every crate. The engine only ever READS
//! through it (its own writes and deletes go through
//! [`JammiObjectStore`](super::JammiObjectStore), which holds the guards), so
//! what is registered there is a [`ReadView`]: it forwards reads and listings
//! to the real driver and refuses every mutation. Whatever route reaches the
//! registry, the store it yields cannot write, overwrite, copy, rename or
//! delete a key under a catalog-managed root.

use std::fmt::{self, Debug, Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::runtime_env::RuntimeEnv;
use futures::stream::{BoxStream, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as StoreResult,
};

use super::url::{Scheme, StorageUrl};
use crate::error::{JammiError, Result};

/// A read-only view of an object store. See the module doc.
#[derive(Debug)]
pub struct ReadView {
    driver: Arc<dyn ObjectStore>,
}

impl ReadView {
    /// A read-only view of `driver`.
    pub fn of(driver: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(Self { driver })
    }

    fn refused(operation: &'static str) -> object_store::Error {
        object_store::Error::NotSupported {
            source: format!(
                "`{operation}` through the query engine's object-store registry: that registry \
                 serves scans and is read-only; the engine's writes and deletes go through its \
                 own guarded storage handle"
            )
            .into(),
        }
    }
}

impl Display for ReadView {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ReadView({})", self.driver)
    }
}

#[async_trait]
impl ObjectStore for ReadView {
    async fn put_opts(&self, _: &Path, _: PutPayload, _: PutOptions) -> StoreResult<PutResult> {
        Err(Self::refused("put"))
    }

    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        Err(Self::refused("put_multipart"))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        self.driver.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> StoreResult<Vec<Bytes>> {
        self.driver.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        locations.map(|_| Err(Self::refused("delete"))).boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.driver.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.driver.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.driver.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> StoreResult<()> {
        Err(Self::refused("copy"))
    }

    async fn rename_opts(&self, _: &Path, _: &Path, _: RenameOptions) -> StoreResult<()> {
        Err(Self::refused("rename"))
    }
}

/// Make `driver` the store DataFusion resolves `url` to — as a [`ReadView`].
/// Registering the engine's own driver (rather than letting DataFusion build
/// one) is what gives a cloud scan the engine's credentials.
///
/// `file://` needs no entry of its own: [`register_local_read_view`] covers
/// the whole scheme. `memory://` is a test-only scheme no scan is driven
/// through.
pub(crate) fn register_read_view(
    runtime: &RuntimeEnv,
    url: &StorageUrl,
    driver: Arc<dyn ObjectStore>,
) -> Result<()> {
    if matches!(url.scheme(), Scheme::File | Scheme::Memory) {
        return Ok(());
    }
    let parsed = ::url::Url::parse(url.as_str())
        .map_err(|e| JammiError::Config(format!("Storage URL '{url}' did not re-parse: {e}")))?;
    runtime.register_object_store(&parsed, ReadView::of(driver));
    Ok(())
}

/// Replace DataFusion's pre-registered `file://` store — a read-write
/// `LocalFileSystem` rooted at `/` — with a read-only view of the same.
pub(crate) fn register_local_read_view(runtime: &RuntimeEnv) -> Result<()> {
    let root = ::url::Url::parse("file://")
        .map_err(|e| JammiError::Config(format!("the file:// root did not parse: {e}")))?;
    runtime.register_object_store(
        &root,
        ReadView::of(Arc::new(object_store::local::LocalFileSystem::new())),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt;

    async fn seeded() -> (Arc<dyn ObjectStore>, Arc<dyn ObjectStore>, Path) {
        let driver: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let key = Path::from("models/tenant/job/adapter.safetensors");
        driver
            .put(&key, PutPayload::from_static(b"weights"))
            .await
            .expect("seed through the real driver");
        (Arc::clone(&driver), ReadView::of(driver), key)
    }

    fn is_refusal(error: &object_store::Error) -> bool {
        matches!(error, object_store::Error::NotSupported { .. })
    }

    /// Reads and listings pass through to the driver unchanged.
    #[tokio::test]
    async fn a_view_reads_what_the_driver_holds() {
        let (_driver, view, key) = seeded().await;
        let bytes = view.get(&key).await.unwrap().bytes().await.unwrap();
        assert_eq!(&bytes[..], b"weights");
        let listed: Vec<_> = view.list(None).try_collect().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].location, key);
    }

    /// Every mutation is refused, and the driver's bytes are untouched.
    #[tokio::test]
    async fn a_view_refuses_every_mutation() {
        let (driver, view, key) = seeded().await;
        let other = Path::from("models/tenant/job/copy.safetensors");

        assert!(is_refusal(&view.delete(&key).await.unwrap_err()));
        assert!(is_refusal(
            &view
                .put(&key, PutPayload::from_static(b"overwritten"))
                .await
                .unwrap_err()
        ));
        assert!(is_refusal(&view.put_multipart(&other).await.unwrap_err()));
        assert!(is_refusal(&view.copy(&key, &other).await.unwrap_err()));
        assert!(is_refusal(&view.rename(&key, &other).await.unwrap_err()));

        let bytes = driver.get(&key).await.unwrap().bytes().await.unwrap();
        assert_eq!(&bytes[..], b"weights");
        assert!(driver.head(&other).await.is_err());
    }
}
