# Changelog

All notable changes to the Jammi AI workspace are recorded here. The
workspace ships every publishable crate at the same
`workspace.package.version`; PyPI `jammi-ai` mirrors that version.

## v0.6.0 — 2026-05-25

### Breaking

- `jammi-engine` crate renamed to `jammi-db`. The crate's description
  was retold to reflect what it actually is: a vector database, SQL
  federation, mutable companion tables, and trigger broker. Update
  Cargo deps:

  ```toml
  # before
  jammi-engine = { version = "0.5.9", ... }
  # after
  jammi-db = { version = "0.6.0", ... }
  ```

  And every Rust use-path:

  ```rust
  // before
  use jammi_engine::{config::JammiConfig, session::JammiSession};
  // after
  use jammi_db::{config::JammiConfig, session::JammiSession};
  ```

- Python import: `import jammi` → `import jammi_ai`. The importable
  module name now matches the PyPI distribution name (`jammi-ai`).

  ```python
  # before
  import jammi
  db = jammi.connect(...)
  # after
  import jammi_ai
  db = jammi_ai.connect(...)
  ```

- The PyO3 native extension path moved from `jammi._native` to
  `jammi_ai._native`. No customer-facing impact unless you were
  reaching past the public `__init__.py` to import internals
  directly.

### Migration

```bash
# Cargo deps and Rust source
sed -i 's/jammi-engine/jammi-db/g; s/jammi_engine/jammi_db/g' \
  Cargo.toml $(find . -name '*.rs')

# Python imports
sed -i 's/^import jammi$/import jammi_ai/; s/^from jammi import/from jammi_ai import/' \
  $(find . -name '*.py')

# Python jammi.X attribute access
sed -i 's/jammi\.connect/jammi_ai.connect/g' $(find . -name '*.py')
```

### Notes

- The `0.5.x` line of releases continues to ship the old names on
  crates.io (`jammi-engine`) and PyPI (`jammi-ai<0.6` carries
  `import jammi`). Pinned consumers keep working; no
  backwards-compatibility shim is provided in `0.6.0` per the
  no-shims engineering rule.
- This is a pure rename. No behavioural changes, no schema migrations,
  no on-disk artifact-format changes.
- The DataFusion catalog schema name (`jammi`), the runtime
  `directories::ProjectDirs` identifier (`ai/jammi/jammi`), the
  protobuf wire packages (`jammi.v1.session`, `jammi.v1.trigger`), and
  the `jammi` CLI binary name are unchanged — those are runtime
  identifiers, not Rust crate or Python module names.
