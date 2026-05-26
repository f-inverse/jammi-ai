# 2. Connect

```python
import jammi_ai

db = jammi_ai.connect(artifact_dir="/tmp/jammi-quickstart", gpu_device=-1)
```

`jammi_ai.connect` returns a `Database` handle backed by a shared tokio
runtime and a SQLite catalog rooted at `artifact_dir`. Everything you do
through this handle — registered sources, embedding tables, mutable
companions, trigger topics — is persisted under that directory.

## Arguments worth knowing

- `artifact_dir` — root for the catalog DB and result Parquet files. Pass
  an empty directory the first time; subsequent runs reuse what's there.
- `gpu_device` — `-1` forces CPU, omit to auto-select GPU 0 when available,
  pass a positive integer to pin a specific device.
- `inference_batch_size` — defaults are tuned for production batch sizes;
  override for tiny workloads (the quickstart uses 8).

## Tenant scoping

`db.with_tenant("tenant-a")` binds every subsequent call to a tenant. The
quickstart runs unscoped (catalog rows have `tenant_id IS NULL`), which is
the right default for a local interactive workflow.

Next: [register a data source](./03_register_source.md).
