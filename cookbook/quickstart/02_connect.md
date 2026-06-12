# 2. Connect

```python
import jammi_ai

db = jammi_ai.connect("file:///tmp/jammi-quickstart")
```

`jammi_ai.connect(target)` is the one front door. A `file://` target returns a
`Database` handle backed by a shared tokio runtime and a SQLite catalog rooted
at the target's path; everything you do through it — registered sources,
embedding tables, mutable companions, trigger topics — is persisted under that
directory. Flip the target to `https://host` or `grpc://host:8081` — no code
change — and the same `connect(target)` returns a remote handle over the
bundled `jammi-client`.

## The target

- `file:///abs/path` — an embedded, in-process engine rooted at the path. Pass
  an empty directory the first time; subsequent runs reuse what's there.
- `https://host` / `grpcs://host:8081` / `http://host` / `grpc://host:8081` —
  a remote engine over the `jammi.v1` gRPC wire.

## Engine tuning is configuration, not arguments

Device and batch size are engine *config*, read from the environment (or a
`JAMMI_CONFIG` TOML), so they apply identically whether the engine runs
in-process or behind a server:

- `JAMMI_GPU__DEVICE` — `-1` forces CPU, `0`+ pins a device (auto-selects GPU 0
  when available otherwise).
- `JAMMI_ENGINE__BATCH_SIZE` — defaults are tuned for production batch sizes;
  override for tiny workloads (the quickstart uses 8).

## Tenant scoping

`db.set_tenant("tenant-a")` binds every subsequent call to a tenant; for a
block-scoped binding that restores the prior tenant on exit, use
`with db.tenant_scope("tenant-a"): ...`. The quickstart runs unscoped (catalog
rows have `tenant_id IS NULL`), which is the right default for a local
interactive workflow.

Next: [register a data source](./03_register_source.md).
