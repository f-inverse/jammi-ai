//! The layered config tree: a TOML file layer deep-merged with a `JAMMI_*`
//! env-var layer, deserialized directly into `JammiConfig` (and every nested
//! type) through a hand-written [`serde::Deserializer`] implementation.
//!
//! # Why a custom `Deserializer` instead of two passes
//!
//! A "parse the file, then patch fields from env" two-pass loader (the shape
//! `apply_env_overrides` used before this module) has to hand-list every
//! overridable field and cannot express "this externally-tagged enum section
//! is selected by an env var, but its payload still needs the file's
//! sibling keys" or "an unknown `JAMMI_CATALOG__POSTGRES__BOGUS` must be
//! refused, not silently dropped". [`Node`] merges the two layers into one
//! tree first — a table merges recursively, and a leaf simply overrides a
//! leaf below it — then deserializes the WHOLE typed config from that one
//! tree in a single pass, so every `#[serde(deny_unknown_fields)]`, every
//! externally-tagged enum, and every nested struct gets env-override support
//! for free, with no per-field hand-listing.
//!
//! # The `Override` node and type-directed lowering (H1)
//!
//! A file table and an env leaf can legitimately disagree on SHAPE at one
//! path — the file spells out `[catalog.postgres]` with its fields, and the
//! env layer spells a bare `JAMMI_CATALOG=postgres`. The merge itself never
//! decides how to resolve that: it is type-blind and simply records
//! [`Node::Override`]. Only the target type's own `deserialize_*` call
//! knows what to do with it:
//!
//! - at an **enum position** ([`Node::deserialize_enum`]), a bare env
//!   variant name over a file table is *lowered*: the file's payload for
//!   THAT SAME variant survives, so `[storage.cloud.s3] region = "…"` plus
//!   `JAMMI_STORAGE__CLOUD=s3` keeps `region`;
//! - at every other position (scalar, string, seq, map, struct), the
//!   upper (env) layer simply wins outright — except a **struct** position,
//!   where an env leaf cannot stand for a whole struct (a struct is set
//!   field-by-field), so that specific combination is a typed error naming
//!   the variable.
//!
//! # Laziness (D1/T4)
//!
//! An env leaf never eagerly guesses its type: `deserialize_bool`/`i*`/`u*`/
//! `f*` parse `raw` as that primitive, `deserialize_str`/`string`/`any`
//! hand back `raw` verbatim (never sniffed as a number or bool), and
//! `deserialize_seq`/`map`/`struct` parse `raw` as a standalone TOML
//! document (so `JAMMI_INFERENCE__HTTP__HEADERS='{ X-Api-Key = "v" }'`
//! parses as a map). `deserialize_enum` treats the raw string as the
//! variant's bare name, with an empty payload for a struct or newtype
//! variant (X4) — never a panic, a `missing field` error surfaces exactly as
//! it would for a file table missing the same key.
//!
//! # No error ever echoes a VALUE (phase-4 audit)
//!
//! An error derived from an env or file VALUE never includes that value —
//! only the struct path, the `JAMMI_*` variable name, the expected
//! type/variant list, and, for a TOML syntax error, a line:column locator.
//! This is enforced by two shared helpers rather than left to each
//! `deserialize_*` arm to get right independently: [`describe_toml_error`]
//! renders a `toml::de::Error` via `message()` + a line:column computed from
//! `span()`, never `Display` (which renders a code frame quoting the
//! offending source line verbatim — the exact shape that would print an
//! env-interpolated secret or a malformed env leaf's raw text); and every
//! `Node::File`/parsed-`Node::Env` arm that forwards a `toml::Value`
//! deserialize failure routes it through [`safe_type_error`], which keeps
//! only a fixed "expected `<shape>`" phrase for serde's `invalid_type`
//! shape (the one shape that bakes the OFFENDING VALUE into the message
//! text with no way to strip it after the fact:
//! `invalid type: string "…", expected …`) and passes every OTHER shape
//! (`missing field …`, `unknown field …`, a nested `Secret`'s own
//! file-read error) through unchanged — those only ever name a field/key
//! NAME or a file PATH, never an arbitrary value, so redacting them would
//! throw away real diagnostic value for no safety gain. An enum position's
//! "unknown variant" error names the variable and the expected variant
//! list — never the value a bare env override supplied when that value
//! does not itself name a variant (`deserialize_enum`'s `Node::Env` arm,
//! and the `Node::Override` enum-lowering arm, which used to synthesize a
//! table keyed by the raw value and let the generic multi-key path re-echo
//! it — see [`Node::deserialize_enum`]'s Override arm for why membership is
//! now checked before that table is ever built, not after).

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{
    self, DeserializeSeed, Deserializer, EnumAccess, IntoDeserializer, MapAccess, VariantAccess,
    Visitor,
};

/// One node of the merged file+env config tree.
#[derive(Debug, Clone)]
pub(crate) enum Node {
    /// A TOML table — from the file layer, an env-built path nesting, or a
    /// merge of both. `env_authored` is true for this table iff it (or any
    /// descendant) was built from at least one env var; [`Node::deserialize_enum`]'s
    /// multi-key provenance resolution reads this bit.
    Table {
        entries: BTreeMap<String, Node>,
        env_authored: bool,
    },
    /// A non-table value from the TOML file layer.
    File(toml::Value),
    /// One `JAMMI_*` variable. `var` is the full name (used in every error
    /// message so a deployer can grep straight to the offending line in
    /// their env); `raw` is its string value, parsed lazily (D1/T4).
    Env { var: String, raw: String },
    /// A lower (file-rooted) and upper (env-rooted) node disagree on shape
    /// at this path. See the module docs for how each `deserialize_*`
    /// resolves this.
    Override { lower: Box<Node>, upper: Box<Node> },
}

/// This `Deserializer` implementation's error type: a plain message, built
/// via `serde::de::Error::custom` the same way every hand-rolled
/// `Deserializer` in the ecosystem builds its error type.
#[derive(Debug)]
pub(crate) struct NodeError(pub String);

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NodeError {}

impl de::Error for NodeError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        NodeError(msg.to_string())
    }
}

impl Node {
    pub(crate) fn env_leaf(var: &str, raw: &str) -> Node {
        Node::Env {
            var: var.to_string(),
            raw: raw.to_string(),
        }
    }

    pub(crate) fn empty_env_table() -> Node {
        Node::Table {
            entries: BTreeMap::new(),
            env_authored: true,
        }
    }

    fn empty_table() -> Node {
        Node::Table {
            entries: BTreeMap::new(),
            env_authored: false,
        }
    }

    fn is_env_authored(&self) -> bool {
        match self {
            Node::Env { .. } => true,
            Node::Table { env_authored, .. } => *env_authored,
            Node::File(_) => false,
            Node::Override { lower, upper } => lower.is_env_authored() || upper.is_env_authored(),
        }
    }

    /// Build a [`Node`] tree from a parsed TOML document: every table
    /// becomes [`Node::Table`] (recursively), every other value becomes
    /// [`Node::File`].
    pub(crate) fn from_toml(value: toml::Value) -> Node {
        match value {
            toml::Value::Table(table) => Node::Table {
                entries: table
                    .into_iter()
                    .map(|(k, v)| (k, Node::from_toml(v)))
                    .collect(),
                env_authored: false,
            },
            other => Node::File(other),
        }
    }

    /// Every `JAMMI_*` variable name reachable under this node — the
    /// provenance list every "refuses naming it/them" error is built from.
    pub(crate) fn vars(&self, out: &mut Vec<String>) {
        match self {
            Node::Env { var, .. } => out.push(var.clone()),
            Node::Table { entries, .. } => {
                for v in entries.values() {
                    v.vars(out);
                }
            }
            Node::File(_) => {}
            Node::Override { lower, upper } => {
                lower.vars(out);
                upper.vars(out);
            }
        }
    }

    /// A short provenance blurb for an error message: the env var name(s)
    /// under this node, or "the config file" when there are none.
    fn origin(&self) -> String {
        let mut vars = Vec::new();
        self.vars(&mut vars);
        if vars.is_empty() {
            "the config file".to_string()
        } else {
            vars.join(", ")
        }
    }

    /// Parse an env value as a standalone TOML document — used by every
    /// seq/map/struct-position env leaf (D1/T4): `JAMMI_...=raw` is parsed
    /// as `key = raw` and the value extracted, so `raw` is read with TOML's
    /// own grammar (quoting, inline tables, arrays) rather than treated as
    /// pre-quoted text.
    fn parse_as_toml(raw: &str) -> Result<toml::Value, NodeError> {
        let doc = format!("v = {raw}");
        let parsed: toml::Value = doc.parse().map_err(|e: toml::de::Error| {
            NodeError(format!(
                "env value is not valid TOML: {}",
                describe_toml_error(&doc, &e)
            ))
        })?;
        Ok(parsed
            .get("v")
            .cloned()
            .expect("the wrapper document always has key `v`"))
    }
}

/// Render a `toml::de::Error` SAFELY: `message()` (the "what went wrong"
/// text alone) plus a 1-based line:column locator computed from `span()`'s
/// start offset against `source` — NEVER `Display`/`to_string()`, which
/// renders a code frame quoting the offending source line verbatim. For a
/// file parse, that source line is the file text AFTER `${VAR}`
/// interpolation — an unquoted `url = ${POSTGRES_URL}` would otherwise
/// print the expanded secret; for an env leaf parsed as a standalone
/// document ([`Node::parse_as_toml`]), that source line IS the (possibly
/// secret-bearing) env value. `source` is whatever text was actually
/// handed to `.parse()` — the caller's job is only to pass the SAME text,
/// so the byte offset lines up.
pub(crate) fn describe_toml_error(source: &str, e: &toml::de::Error) -> String {
    match e.span() {
        Some(span) => {
            let (line, column) = line_and_column(source, span.start);
            format!("{} (line {line}, column {column})", e.message())
        }
        None => e.message().to_string(),
    }
}

/// 1-based (line, column) for a byte offset into `source`, scanning by
/// `char` (never splitting a UTF-8 code point) up to `offset`.
fn line_and_column(source: &str, offset: usize) -> (usize, usize) {
    let bound = offset.min(source.len());
    let mut line = 1usize;
    let mut column = 1usize;
    for ch in source[..bound].chars() {
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

/// Keep only a fixed "expected `<shape>`" phrase for serde's `invalid_type`
/// error shape — the one shape that bakes the OFFENDING VALUE into the
/// message text with no way to strip it after the fact (`invalid type:
/// string "Bearer hunter2-secret", expected a map`) — and pass every OTHER
/// shape through (`missing field …`, `unknown field …`, a nested `Secret`'s
/// own file-read error: each names only a field/key NAME or a file PATH,
/// never an arbitrary value, so redacting them would throw away real
/// diagnostic value for no safety gain). `source` is the exact text that
/// was parsed to produce the value this error came from, when the caller
/// has one to give (an env leaf parsed via [`Node::parse_as_toml`]) — used
/// to compute a safe line:column locator via [`describe_toml_error`] for
/// the passed-through shapes; a `Node::File` arm has no single source text
/// of its own to hand back (the whole file was already parsed once, higher
/// up), so it passes `None` and gets `message()` alone, with no locator —
/// still safe, just less precise.
fn safe_type_error(source: Option<&str>, e: toml::de::Error, expected: &str) -> NodeError {
    if e.message().starts_with("invalid type: ") {
        NodeError(format!("expected {expected}"))
    } else {
        match source {
            Some(src) => NodeError(describe_toml_error(src, &e)),
            None => NodeError(e.message().to_string()),
        }
    }
}

/// Deep-merge `over` onto `base`. Type-blind (H1): a genuine shape
/// disagreement (a table meeting a leaf) is recorded as [`Node::Override`]
/// rather than resolved here, because only the eventual `deserialize_*` call
/// knows whether this is an enum position.
pub(crate) fn merge(base: Node, over: Node) -> Node {
    match (base, over) {
        (
            Node::Table {
                entries: mut base_entries,
                ..
            },
            Node::Table {
                entries: over_entries,
                ..
            },
        ) => {
            for (k, v) in over_entries {
                let merged = match base_entries.remove(&k) {
                    Some(bv) => merge(bv, v),
                    None => v,
                };
                base_entries.insert(k, merged);
            }
            let env_authored = base_entries.values().any(Node::is_env_authored);
            Node::Table {
                entries: base_entries,
                env_authored,
            }
        }
        (base @ Node::Table { .. }, over @ Node::Env { .. }) => Node::Override {
            lower: Box::new(base),
            upper: Box::new(over),
        },
        (base @ (Node::File(_) | Node::Env { .. }), over @ Node::Table { .. }) => Node::Override {
            lower: Box::new(base),
            upper: Box::new(over),
        },
        (_, over) => over,
    }
}

macro_rules! scalar {
    ($method:ident, $visit:ident, $ty:ty) => {
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
            match self {
                Node::Env { ref var, ref raw } => {
                    let value: $ty = raw
                        .trim()
                        .parse()
                        .map_err(|e| NodeError(format!("{var}: {e}")))?;
                    visitor.$visit(value)
                }
                Node::File(v) => v.$method(visitor).map_err(|e| {
                    safe_type_error(None, e, concat!("a `", stringify!($ty), "` value"))
                }),
                Node::Table { .. } => Err(NodeError(format!(
                    "expected a scalar value for `{}`, found a table",
                    stringify!($method)
                ))),
                Node::Override { upper, .. } => upper.$method(visitor),
            }
        }
    };
}

/// The four bool spellings a shell, a container manifest, or an
/// orchestrator template actually emits: `true`/`false`/`1`/`0`,
/// case-insensitive, whitespace-trimmed. Every `bool`-typed config field's
/// env leaf accepts exactly this domain — never `bool::from_str`'s strict
/// `"true"`/`"false"` alone (which would reject the numeric and
/// upper-cased spellings every deployer actually types), and never a silent
/// guess outside it.
pub(crate) fn parse_lenient_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

impl<'de> Deserializer<'de> for Node {
    type Error = NodeError;

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        match self {
            Node::Env { ref var, ref raw } => {
                // Names the variable only — never the value: a bool-typed
                // field can sit at the same path shape a header/credential
                // map does in a differently-typed sibling, and this error
                // must stay safe to paste into a startup log regardless.
                let value = parse_lenient_bool(raw).ok_or_else(|| {
                    NodeError(format!(
                        "{var}: invalid boolean value; expected one of true, false, 1, 0"
                    ))
                })?;
                visitor.visit_bool(value)
            }
            Node::File(v) => v
                .deserialize_bool(visitor)
                .map_err(|e| safe_type_error(None, e, "a boolean value")),
            Node::Table { .. } => Err(NodeError(
                "expected a scalar value for `deserialize_bool`, found a table".into(),
            )),
            Node::Override { upper, .. } => upper.deserialize_bool(visitor),
        }
    }

    scalar!(deserialize_i8, visit_i8, i8);
    scalar!(deserialize_i16, visit_i16, i16);
    scalar!(deserialize_i32, visit_i32, i32);
    scalar!(deserialize_i64, visit_i64, i64);
    scalar!(deserialize_u8, visit_u8, u8);
    scalar!(deserialize_u16, visit_u16, u16);
    scalar!(deserialize_u32, visit_u32, u32);
    scalar!(deserialize_u64, visit_u64, u64);
    scalar!(deserialize_f32, visit_f32, f32);
    scalar!(deserialize_f64, visit_f64, f64);
    scalar!(deserialize_char, visit_char, char);

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        match self {
            Node::Env { ref raw, .. } => visitor.visit_str(raw),
            Node::File(v) => v
                .deserialize_any(visitor)
                .map_err(|e| safe_type_error(None, e, "a value valid for this field")),
            Node::Table { entries, .. } => visitor.visit_map(TableMap {
                iter: entries.into_iter(),
                value: None,
            }),
            Node::Override { upper, .. } => upper.deserialize_any(visitor),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        match self {
            Node::Env { ref raw, .. } => visitor.visit_str(raw),
            Node::File(v) => v
                .deserialize_str(visitor)
                .map_err(|e| safe_type_error(None, e, "a string")),
            Node::Table { .. } => Err(NodeError("expected a string, found a table".into())),
            Node::Override { upper, .. } => upper.deserialize_str(visitor),
        }
    }
    fn deserialize_string<V: Visitor<'de>>(self, v: V) -> Result<V::Value, NodeError> {
        self.deserialize_str(v)
    }
    fn deserialize_bytes<V: Visitor<'de>>(self, v: V) -> Result<V::Value, NodeError> {
        self.deserialize_str(v)
    }
    fn deserialize_byte_buf<V: Visitor<'de>>(self, v: V) -> Result<V::Value, NodeError> {
        self.deserialize_str(v)
    }
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        // The key is present (or `Node::Override` wouldn't have formed and
        // no `Node` would be handed to this position at all), so it means
        // `Some`; the inner type decides how to read `self`.
        visitor.visit_some(self)
    }
    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        visitor.visit_unit()
    }
    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, NodeError> {
        self.deserialize_unit(visitor)
    }
    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, NodeError> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        match self {
            Node::Env { ref raw, .. } => {
                let doc = format!("v = {raw}");
                let v = Node::parse_as_toml(raw)?;
                v.deserialize_seq(visitor)
                    .map_err(|e| safe_type_error(Some(&doc), e, "a sequence (TOML array)"))
            }
            Node::File(v) => v
                .deserialize_seq(visitor)
                .map_err(|e| safe_type_error(None, e, "a sequence (TOML array)")),
            Node::Table { .. } => Err(NodeError("expected a sequence, found a table".into())),
            // H14: a seq position takes the whole upper value — a file
            // array is wholly replaced by an env array, never merged.
            Node::Override { upper, .. } => upper.deserialize_seq(visitor),
        }
    }
    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, v: V) -> Result<V::Value, NodeError> {
        self.deserialize_seq(v)
    }
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        v: V,
    ) -> Result<V::Value, NodeError> {
        self.deserialize_seq(v)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        match self {
            Node::Env { ref raw, .. } => {
                let doc = format!("v = {raw}");
                let v = Node::parse_as_toml(raw)?;
                v.deserialize_map(visitor)
                    .map_err(|e| safe_type_error(Some(&doc), e, "a map (TOML inline table)"))
            }
            Node::File(v) => v
                .deserialize_map(visitor)
                .map_err(|e| safe_type_error(None, e, "a map (TOML inline table)")),
            Node::Table { entries, .. } => visitor.visit_map(TableMap {
                iter: entries.into_iter(),
                value: None,
            }),
            // H14: a map position takes the whole upper value — a file
            // `[a.b]` table is wholly replaced by an env inline table, never
            // merged. (t8.rs's `deserialize_any` dispatch here is
            // superseded by this explicit `deserialize_map`.)
            Node::Override { upper, .. } => upper.deserialize_map(visitor),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, NodeError> {
        // H14: a struct position cannot take a whole-value env override — a
        // struct is set field-by-field from env, never in one variable.
        // This refusal is identical whether or not a lower (file) layer is
        // present at this path: with a file layer the merge records
        // `Node::Override { upper: Env, .. }`; with NO file layer at all
        // (nothing to merge against) the env leaf reaches this function
        // directly as a bare `Node::Env` — that bare case must refuse too,
        // or a whole-struct env value silently resets every sibling field
        // to its default with no file layer to blame it on.
        fn struct_position_env_refusal(var: &str) -> NodeError {
            NodeError(format!(
                "{var}: this is a struct, set field-by-field from env \
                 (e.g. `{var}__FIELD=...`), not as a whole value"
            ))
        }
        match self {
            Node::Env { ref var, .. } => Err(struct_position_env_refusal(var)),
            Node::Override { upper, .. } => match *upper {
                Node::Env { ref var, .. } => Err(struct_position_env_refusal(var)),
                other => other.deserialize_map(visitor),
            },
            other => other.deserialize_map(visitor),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, v: V) -> Result<V::Value, NodeError> {
        self.deserialize_str(v)
    }
    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NodeError> {
        visitor.visit_unit()
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, NodeError> {
        match self {
            Node::Table { entries, .. } => {
                // X2: fail-closed on any key that is not a known variant,
                // even when an env override selects a different, valid one.
                for (k, v) in entries.iter() {
                    if !variants.contains(&k.as_str()) {
                        return Err(NodeError(format!(
                            "{name}: unknown variant `{k}` (from {}); expected one of {variants:?}",
                            v.origin()
                        )));
                    }
                }
                let (key, value) = if entries.len() == 1 {
                    entries.into_iter().next().expect("len == 1")
                } else {
                    let env_keys: Vec<String> = entries
                        .iter()
                        .filter(|(_, v)| v.is_env_authored())
                        .map(|(k, _)| k.clone())
                        .collect();
                    if env_keys.len() == 1 {
                        let mut entries = entries;
                        let key = env_keys.into_iter().next().expect("len == 1");
                        let value = entries.remove(&key).expect("key came from this map");
                        (key, value)
                    } else {
                        let keys: Vec<_> = entries.keys().cloned().collect();
                        let mut vars = Vec::new();
                        for v in entries.values() {
                            v.vars(&mut vars);
                        }
                        return Err(NodeError(format!(
                            "{name}: exactly one of {variants:?} must be set (found {keys:?}; \
                             env vars: {vars:?})"
                        )));
                    }
                };
                visitor.visit_enum(NodeEnum {
                    key,
                    value: Some(value),
                })
            }
            // X4: a bare file string at an enum position behaves exactly
            // like a bare env leaf.
            Node::File(toml::Value::String(s)) => visitor.visit_enum(NodeEnum {
                key: s.trim().to_string(),
                value: None,
            }),
            Node::File(v) => v.deserialize_enum(name, variants, visitor).map_err(|e| {
                safe_type_error(None, e, &format!("a string or table (one of {variants:?})"))
            }),
            Node::Env { ref var, ref raw } => {
                let key = raw.trim().to_string();
                if !variants.contains(&key.as_str()) {
                    // H13 pinned oracle: a bare env selection naming an
                    // unknown variant, with no file layer to blame,
                    // errors naming the VARIABLE — never `key`/`raw`: a bare
                    // enum-position env override that ISN'T a real variant
                    // name is exactly the shape a whole-value payload like
                    // `JAMMI_CATALOG='{ postgres = { url = "…secret…" } }'`
                    // takes, and `key` there is that entire blob, secret
                    // included.
                    return Err(NodeError(format!(
                        "{name}: {var} does not name a known variant; expected one of {variants:?}"
                    )));
                }
                visitor.visit_enum(NodeEnum { key, value: None })
            }
            // H1 + H14: the round-4 type-directed lowering. Only the
            // deserializer (here) knows the variant list, so the merge
            // deferred this decision to us.
            Node::Override { lower, upper } => {
                let lowered = match (*lower, *upper) {
                    // A bare env variant name over a file table: keep the
                    // file's payload for THAT SAME variant. Every key in
                    // the resulting table — winner included — is still
                    // membership-checked below (H13 wording).
                    (
                        Node::Table {
                            entries: mut base, ..
                        },
                        Node::Env { var, raw },
                    ) => {
                        let key = raw.trim().to_string();
                        if !variants.contains(&key.as_str()) {
                            // Same rule as the bare-`Node::Env` arm above,
                            // checked HERE (before `key` ever becomes a
                            // table entry) rather than left to the generic
                            // `Node::Table` arm's per-key membership check:
                            // that check is safe for a genuine file-authored
                            // section name, but `key` here can be an entire
                            // env VALUE (e.g. a whole-value payload with a
                            // secret in it) that merely failed to name a
                            // variant — inserting it as a synthesized table
                            // key and letting the generic path re-discover
                            // "unknown variant" would echo that value, and
                            // would misattribute it to "the config file"
                            // (an empty synthesized table has no env var of
                            // its own for `.origin()` to find).
                            return Err(NodeError(format!(
                                "{name}: {var} does not name a known variant; expected one of {variants:?}"
                            )));
                        }
                        let sub = match base.remove(&key) {
                            Some(Node::Table { entries, .. }) => Node::Table {
                                entries,
                                env_authored: true,
                            },
                            Some(other) => other,
                            None => Node::empty_env_table(),
                        };
                        base.insert(key, sub);
                        Node::Table {
                            entries: base,
                            env_authored: true,
                        }
                    }
                    // A bare file-string selection under a nested env
                    // table: the env table wins, but the FILE's bare
                    // spelling is still membership-checked below (via the
                    // synthesized entry inserted under `base`), so a
                    // typo'd file selection is still named in the error
                    // rather than silently discarded.
                    (Node::File(toml::Value::String(s)), Node::Table { entries: over, .. }) => {
                        let mut base: BTreeMap<String, Node> = BTreeMap::new();
                        base.insert(s.trim().to_string(), Node::empty_table());
                        for (k, v) in over {
                            let merged = match base.remove(&k) {
                                Some(bv) => merge(bv, v),
                                None => v,
                            };
                            base.insert(k, merged);
                        }
                        Node::Table {
                            entries: base,
                            env_authored: true,
                        }
                    }
                    (lo, up) => {
                        return Err(NodeError(format!(
                            "{name}: conflicting selection at an enum position between {} and {}",
                            lo.origin(),
                            up.origin()
                        )));
                    }
                };
                lowered.deserialize_enum(name, variants, visitor)
            }
        }
    }
}

struct NodeEnum {
    key: String,
    value: Option<Node>,
}

impl<'de> EnumAccess<'de> for NodeEnum {
    type Error = NodeError;
    type Variant = NodeVariant;

    fn variant_seed<S: DeserializeSeed<'de>>(
        self,
        seed: S,
    ) -> Result<(S::Value, NodeVariant), NodeError> {
        let key = seed.deserialize(self.key.into_deserializer())?;
        Ok((key, NodeVariant(self.value)))
    }
}

struct NodeVariant(Option<Node>);

impl<'de> VariantAccess<'de> for NodeVariant {
    type Error = NodeError;

    // X3: a unit variant refuses a non-empty payload.
    fn unit_variant(self) -> Result<(), NodeError> {
        match self.0 {
            None => Ok(()),
            Some(Node::Table { entries, .. }) if entries.is_empty() => Ok(()),
            Some(Node::Table { entries, .. }) => {
                let keys: Vec<_> = entries.keys().cloned().collect();
                let mut vars = Vec::new();
                for v in entries.values() {
                    v.vars(&mut vars);
                }
                Err(NodeError(format!(
                    "this variant takes no options; unexpected key(s) {keys:?} (env vars: {vars:?})"
                )))
            }
            Some(other) => Err(NodeError(format!(
                "this variant takes no options (from {})",
                other.origin()
            ))),
        }
    }

    // X4: a bare selection (no payload) deserializes as if it named an
    // empty table — a struct/newtype payload of all-optional fields parses
    // to its defaults, and one with a required field surfaces the ordinary
    // `missing field` error, never a panic.
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, NodeError> {
        seed.deserialize(self.0.unwrap_or_else(Node::empty_table))
    }
    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, v: V) -> Result<V::Value, NodeError> {
        match self.0 {
            Some(n) => n.deserialize_seq(v),
            None => Err(NodeError("this tuple variant needs a payload".into())),
        }
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, NodeError> {
        // Route through `deserialize_struct` (not `deserialize_map`
        // directly) so the H14 Override/struct-position rule applies
        // uniformly whether the struct is a bare field or an enum
        // variant's payload.
        self.0
            .unwrap_or_else(Node::empty_table)
            .deserialize_struct("", fields, visitor)
    }
}

struct TableMap {
    iter: std::collections::btree_map::IntoIter<String, Node>,
    value: Option<Node>,
}

impl<'de> MapAccess<'de> for TableMap {
    type Error = NodeError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, NodeError> {
        match self.iter.next() {
            Some((k, v)) => {
                self.value = Some(v);
                seed.deserialize(k.into_deserializer()).map(Some)
            }
            None => Ok(None),
        }
    }
    fn next_value_seed<S: DeserializeSeed<'de>>(&mut self, seed: S) -> Result<S::Value, NodeError> {
        seed.deserialize(
            self.value
                .take()
                .expect("next_value_seed called after next_key_seed"),
        )
    }
}

// `SeqAccess` is never implemented directly on `Node`: every seq position is
// served by delegating to `toml::Value`'s own `Deserializer` impl (a `File`
// leaf) or by parsing an env/raw string as TOML first
// (`Node::parse_as_toml`), both of which hand the visitor a `toml::Value`
// seq, never a bare `Node` seq.
