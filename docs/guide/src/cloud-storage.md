# Store Sources and Results in Cloud Object Storage

Jammi treats local disk, S3, GCS, Azure Blob, and Cloudflare R2 as interchangeable backends. Any place the engine accepts a local file path it also accepts a storage URL — `file://`, `s3://`, `gs://`, `azure://`, or `r2://` — including registered file-shaped sources and the result-table Parquet that embedding and inference jobs write.

## Build with the cloud features you need

The default build ships only `file://` and the in-memory test driver. Cloud schemes are opt-in per provider so a deployment that only uses S3 does not pull in the GCS and Azure SDK chains:

| Feature | Schemes it enables |
|---------|--------------------|
| `storage-s3` | `s3://` (AWS S3 and S3-compatible: MinIO, LocalStack) |
| `storage-gcs` | `gs://` |
| `storage-azure` | `azure://`, `abfss://` |
| `storage-r2` | `r2://` (Cloudflare R2 — the S3 driver with R2's endpoint + region derived) |
| `storage-cloud` | All four (umbrella) |

```toml
[dependencies]
jammi-db = { version = "0.5", features = ["storage-s3", "storage-gcs"] }
```

Live integration tests live behind matching `live-s3-tests`, `live-gcs-tests`, `live-azure-tests` features so the hermetic `cargo test` lane never reaches the network.

## Register an S3-backed source

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> Result<(), Box<dyn std::error::Error>> {
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{CloudConfig, S3Config, StorageUrl};

let url = StorageUrl::parse("s3://benchmarks/snapshots/2026/papers.parquet")?;

let conn = SourceConnection {
    url: Some(url.to_string()),
    format: Some(FileFormat::Parquet),
    cloud: Some(CloudConfig::S3(S3Config {
        region: Some("us-east-1".into()),
        ..Default::default()
    })),
    ..Default::default()
};

session.add_source("papers", SourceType::File, conn).await?;

let rows = session
    .sql("SELECT id, title FROM papers.public.papers LIMIT 10")
    .await?;
# Ok(()) }
```

If the `cloud` field is `None` and the URL is a cloud scheme, the driver falls back to the SDK's ambient credential chain — env vars, instance profile, IRSA, ADC, Managed Identity.

### Python

```python
from jammi_ai import Database

db = Database()
db.add_source("papers", url="s3://benchmarks/snapshots/2026/papers.parquet", format="parquet")
db.sql("SELECT id, title FROM papers.public.papers LIMIT 10")
```

The Python binding accepts the same URL forms as the Rust API; per-source cloud credentials are read from process environment.

### CLI

```bash
jammi sources add papers \
    --url s3://benchmarks/snapshots/2026/papers.parquet \
    --format parquet
```

## GCS and Azure

The pattern is identical — only the URL prefix and the `CloudConfig` variant change:

```rust,no_run
# extern crate jammi_db;
# use jammi_db::source::{FileFormat, SourceConnection};
# fn make() -> SourceConnection {
use jammi_db::storage::{CloudConfig, GcsConfig};

let conn = SourceConnection {
    url: Some("gs://archives/2026/jan.parquet".into()),
    format: Some(FileFormat::Parquet),
    cloud: Some(CloudConfig::Gcs(GcsConfig {
        service_account_path: Some("/etc/jammi/sa.json".into()),
        ..Default::default()
    })),
    ..Default::default()
};
# conn }
```

```rust,no_run
# extern crate jammi_db;
# use jammi_db::source::{FileFormat, SourceConnection};
# fn make() -> Result<SourceConnection, Box<dyn std::error::Error>> {
use jammi_db::storage::{AzureConfig, CloudConfig};

let conn = SourceConnection {
    url: Some("azure://snapshots/model_outputs.parquet".into()),
    format: Some(FileFormat::Parquet),
    cloud: Some(CloudConfig::Azure(AzureConfig {
        account_name: Some("mystorage".into()),
        sas_token: Some(std::env::var("AZURE_SAS_TOKEN")?),
        ..Default::default()
    })),
    ..Default::default()
};
# Ok(conn) }
```

## Cloudflare R2

R2 speaks the S3 API, so it rides the same driver — but `r2://` is a first-class scheme so you supply only the R2-shaped inputs and the engine derives the two quirks R2 imposes: the account-scoped endpoint `https://<account_id>.r2.cloudflarestorage.com` and `region = "auto"`. Mint an S3-style access key pair in the R2 dashboard (or via the API) and give Jammi the account id:

```rust,no_run
# extern crate jammi_db;
# use jammi_db::source::{FileFormat, SourceConnection};
# fn make() -> Result<SourceConnection, Box<dyn std::error::Error>> {
use jammi_db::storage::{CloudConfig, R2Config};

let conn = SourceConnection {
    url: Some("r2://archives/snapshots/2026.parquet".into()),
    format: Some(FileFormat::Parquet),
    cloud: Some(CloudConfig::R2(R2Config {
        account_id: Some(std::env::var("R2_ACCOUNT_ID")?),
        access_key_id: Some(std::env::var("R2_ACCESS_KEY_ID")?),
        secret_access_key: Some(std::env::var("R2_SECRET_ACCESS_KEY")?),
        ..Default::default()
    })),
    ..Default::default()
};
# Ok(conn) }
```

Set `endpoint` instead of `account_id` to point at an R2 custom domain. Result tables and their sidecar ANN indexes persist to `r2://` exactly as to any other cloud backend.

## Persist result tables to the cloud

`ResultStore` accepts a [`StorageUrl`] root, so embedding and inference outputs land in the same bucket as the source data:

```rust,no_run
# extern crate jammi_db;
# use std::sync::Arc;
# use jammi_db::catalog::Catalog;
# fn ex(catalog: Arc<Catalog>) -> jammi_db::error::Result<()> {
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::ResultStore;
use std::sync::Arc;

let root = StorageUrl::parse("s3://benchmarks/jammi_db")?;
let registry = StorageRegistry::new();
let result_store = Arc::new(ResultStore::with_root(root, registry, catalog)?);
# Ok(()) }
```

Every result table the session creates writes its Parquet and sidecar ANN index to that prefix; `delete_table_files` and the crash-recovery pass operate against the same backend.

## How the layout maps onto buckets

For a result table named `papers__text_embedding__bge-m3__20260520T120000Z_abc12345`, the engine writes three siblings:

```text
s3://benchmarks/jammi_db/papers__text_embedding__bge-m3__….parquet
s3://benchmarks/jammi_db/papers__text_embedding__bge-m3__….idx.usearch
s3://benchmarks/jammi_db/papers__text_embedding__bge-m3__….idx.rowmap
s3://benchmarks/jammi_db/papers__text_embedding__bge-m3__….idx.manifest.json
```

The sidecar layout is the same on every backend; the only difference is the driver under the hood. USearch's path-based FFI is bridged through a tempfile for cloud schemes so its `save` / `load` calls work unchanged.
