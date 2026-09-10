//! `JAMMI_*` environment variables → the env-layer [`Node`] tree, and the
//! namespace rule (T5) that decides which `JAMMI_*` variables are config at
//! all.
//!
//! # The namespace rule
//!
//! - `JAMMI_<X>__<path>` is **always** config: an unknown `X` (not one of
//!   [`TOP_LEVEL_FIELDS`]) is a typed error naming the variable, never a
//!   silent no-op. This is esc-095's fix: `JAMMI_CATALOG__KIND=postgres` — a
//!   typo one segment short of `JAMMI_CATALOG__POSTGRES__…` — refuses rather
//!   than running SQLite with nothing to explain why.
//! - `JAMMI_<X>` with no `__` is config **iff** `X` exactly names a
//!   top-level `JammiConfig` field (e.g. `JAMMI_ARTIFACT_DIR`). Every other
//!   `JAMMI_*` name — and everything without the `JAMMI_` prefix — is a
//!   runtime knob outside this layer's namespace (`JAMMI_AUDIT_MASTER_KEY`,
//!   `JAMMI_CONFIG`, `JAMMI_TEST_PG_URL`, `JAMMI_KERNELS_DISABLE`, …) and is
//!   silently ignored here, exactly as it always was.
//!
//! Path segments (after the first) are lowercased on the way into the tree,
//! matching every config struct's `snake_case` field names; a map key
//! reached via a path segment is therefore always lowercased, while a value
//! written as a TOML inline table (`JAMMI_INFERENCE__HTTP__HEADERS='{ X-Api-Key
//! = "v" }'`) preserves the case of its own keys (R8) because those keys
//! never pass through this segment-splitting logic at all — they are parsed
//! as TOML by [`super::layers::Node`]'s lazy env leaf.

use std::collections::BTreeMap;
use std::fmt;

use super::layers::Node;

/// The exact top-level `JammiConfig` field names a bare `JAMMI_<X>` (no
/// `__`) may select. Keep this set-equal to `JammiConfig`'s fields —
/// `top_level_fields_matches_jammi_config` in `tests.rs` pins it.
pub(crate) const TOP_LEVEL_FIELDS: &[&str] = &[
    "artifact_dir",
    "engine",
    "gpu",
    "inference",
    "embedding",
    "fine_tuning",
    "lease",
    "training",
    "cache",
    "server",
    "logging",
    "catalog",
    "broker",
    "signing_key",
    "storage",
    "models",
];

#[derive(Debug)]
pub(crate) struct EnvMapError(pub String);

impl fmt::Display for EnvMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for EnvMapError {}

/// Build the merged env-var [`Node::Table`] from `env`, applying the
/// namespace rule above. `env` need not be pre-filtered to `JAMMI_*` —
/// everything else (and everything under the `JAMMI_` prefix that fails the
/// namespace rule's "runtime knob" branch) is silently skipped.
pub(crate) fn build_env_layer<I>(env: I) -> Result<Node, EnvMapError>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut root: BTreeMap<String, Node> = BTreeMap::new();
    for (var, raw) in env {
        let Some(rest) = var.strip_prefix("JAMMI_") else {
            continue;
        };
        if rest.is_empty() {
            continue;
        }
        if let Some(head_end) = rest.find("__") {
            let head = rest[..head_end].to_lowercase();
            if !TOP_LEVEL_FIELDS.contains(&head.as_str()) {
                return Err(EnvMapError(format!(
                    "{var}: unknown top-level config section `{head}`; expected one of \
                     {TOP_LEVEL_FIELDS:?}"
                )));
            }
            let segments: Vec<String> = rest.split("__").map(str::to_lowercase).collect();
            insert(&mut root, &segments, &var, &raw)?;
        } else {
            let field = rest.to_lowercase();
            if TOP_LEVEL_FIELDS.contains(&field.as_str()) {
                insert(&mut root, std::slice::from_ref(&field), &var, &raw)?;
            }
            // Else: a runtime knob outside the config namespace. Ignored —
            // this is the branch that keeps `JAMMI_AUDIT_MASTER_KEY`,
            // `JAMMI_CONFIG`, `JAMMI_TEST_PG_URL`, `JAMMI_KERNELS_DISABLE`,
            // and friends inert here.
        }
    }
    Ok(Node::Table {
        entries: root,
        env_authored: true,
    })
}

/// Insert one `var=raw` pair at `segments` into `map`, applying the T6
/// order-independent leaf/table collision rule: a variable that would set a
/// value at a path another variable already nests under (or vice versa) is
/// a typed error naming BOTH variables, regardless of insertion order —
/// because the error is raised the moment the second one collides with
/// whatever the first already placed, symmetrically for either arm.
fn insert(
    map: &mut BTreeMap<String, Node>,
    segments: &[String],
    var: &str,
    raw: &str,
) -> Result<(), EnvMapError> {
    let (head, tail) = segments
        .split_first()
        .expect("segments is never empty: the namespace rule requires at least one");
    if tail.is_empty() {
        match map.get(head) {
            Some(Node::Table { entries, .. }) => {
                let nested = Node::Table {
                    entries: entries.clone(),
                    env_authored: true,
                };
                let mut nested_vars = Vec::new();
                nested.vars(&mut nested_vars);
                return Err(EnvMapError(format!(
                    "{var} sets a value at a path that {nested_vars:?} also nests under"
                )));
            }
            Some(Node::Env { var: other, .. }) => {
                return Err(EnvMapError(format!("{var} and {other} set the same path")));
            }
            _ => {}
        }
        map.insert(head.clone(), Node::env_leaf(var, raw));
        Ok(())
    } else {
        match map.get_mut(head) {
            Some(Node::Table { entries, .. }) => insert(entries, tail, var, raw),
            Some(Node::Env { var: other, .. }) => Err(EnvMapError(format!(
                "{var} nests under a path {other} already sets as a value"
            ))),
            Some(Node::File(_)) | Some(Node::Override { .. }) => {
                unreachable!("the env layer alone never contains File/Override nodes")
            }
            None => {
                let mut sub = BTreeMap::new();
                insert(&mut sub, tail, var, raw)?;
                map.insert(
                    head.clone(),
                    Node::Table {
                        entries: sub,
                        env_authored: true,
                    },
                );
                Ok(())
            }
        }
    }
}
