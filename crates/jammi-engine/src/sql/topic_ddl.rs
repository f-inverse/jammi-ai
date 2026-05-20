//! Minimal parser for the Jammi-specific Flight SQL DDL statements
//! `CREATE TOPIC` and `DROP TOPIC`. Recognises the surface SPEC-04 §4 pins;
//! a non-DDL SQL string returns `None` and the caller routes it to
//! DataFusion unchanged.

use std::collections::BTreeMap;

use arrow_schema::{DataType, Field, Schema};

/// Outcome of inspecting one SQL string for trigger-stream DDL.
#[derive(Debug, Clone)]
pub enum TopicDdl {
    /// `CREATE TOPIC <name> (...) [WITH (...)]`.
    Create(CreateTopic),
    /// `DROP TOPIC [IF EXISTS] <name>`.
    Drop(DropTopic),
}

#[derive(Debug, Clone)]
pub struct CreateTopic {
    pub name: String,
    pub schema: Schema,
    pub broker_metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct DropTopic {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum TopicDdlError {
    #[error("topic DDL parse failure: {0}")]
    Parse(String),
}

/// Inspect `sql` for a `CREATE TOPIC` or `DROP TOPIC` statement. Returns
/// `Ok(None)` for any input that is not trigger-stream DDL — those flow
/// straight through to DataFusion. `Ok(Some(_))` carries the parsed DDL.
pub fn maybe_parse(sql: &str) -> Result<Option<TopicDdl>, TopicDdlError> {
    let trimmed = strip_trailing_semicolon(sql.trim());
    let upper = trimmed.to_uppercase();
    if upper.starts_with("CREATE TOPIC") {
        Ok(Some(TopicDdl::Create(parse_create_topic(trimmed)?)))
    } else if upper.starts_with("DROP TOPIC") {
        Ok(Some(TopicDdl::Drop(parse_drop_topic(trimmed)?)))
    } else {
        Ok(None)
    }
}

fn strip_trailing_semicolon(s: &str) -> &str {
    let s = s.trim_end();
    s.strip_suffix(';').map(str::trim_end).unwrap_or(s)
}

fn parse_create_topic(input: &str) -> Result<CreateTopic, TopicDdlError> {
    // `CREATE TOPIC <name> ( col_spec, col_spec, ... ) [WITH ( k = 'v', ... )]`
    let after_keyword = input
        .strip_prefix("CREATE TOPIC")
        .or_else(|| input.strip_prefix("create topic"))
        .or_else(|| {
            // Case-insensitive strip.
            let len = "CREATE TOPIC".len();
            if input.len() >= len && input[..len].eq_ignore_ascii_case("CREATE TOPIC") {
                Some(&input[len..])
            } else {
                None
            }
        })
        .ok_or_else(|| TopicDdlError::Parse("expected `CREATE TOPIC`".into()))?
        .trim_start();

    let (name, rest) = take_identifier(after_keyword)?;
    let rest = rest.trim_start();

    let rest = rest
        .strip_prefix('(')
        .ok_or_else(|| TopicDdlError::Parse("expected `(` after topic name".into()))?;
    let (cols_text, after_cols) = take_balanced_parens(rest)?;
    let fields = parse_column_list(cols_text)?;

    let after_cols = after_cols.trim_start();
    let broker_metadata = if after_cols.is_empty() {
        BTreeMap::new()
    } else if let Some(after_with) = strip_keyword(after_cols, "WITH") {
        let after_with = after_with.trim_start();
        let after_open = after_with
            .strip_prefix('(')
            .ok_or_else(|| TopicDdlError::Parse("expected `(` after `WITH`".into()))?;
        let (opts_text, tail) = take_balanced_parens(after_open)?;
        let tail = tail.trim();
        if !tail.is_empty() {
            return Err(TopicDdlError::Parse(format!(
                "unexpected trailing tokens: {tail}"
            )));
        }
        parse_with_options(opts_text)?
    } else {
        return Err(TopicDdlError::Parse(format!(
            "unexpected tokens after column list: {after_cols}"
        )));
    };

    Ok(CreateTopic {
        name,
        schema: Schema::new(fields),
        broker_metadata,
    })
}

fn parse_drop_topic(input: &str) -> Result<DropTopic, TopicDdlError> {
    let rest = strip_keyword(input, "DROP")
        .ok_or_else(|| TopicDdlError::Parse("expected `DROP`".into()))?
        .trim_start();
    let rest = strip_keyword(rest, "TOPIC")
        .ok_or_else(|| TopicDdlError::Parse("expected `TOPIC`".into()))?
        .trim_start();
    let (if_exists, rest) = match strip_keyword(rest, "IF") {
        Some(after_if) => {
            let after_if = after_if.trim_start();
            let after_exists = strip_keyword(after_if, "EXISTS").ok_or_else(|| {
                TopicDdlError::Parse("expected `IF EXISTS` or just `IF`-less name".into())
            })?;
            (true, after_exists.trim_start())
        }
        None => (false, rest),
    };
    let (name, tail) = take_identifier(rest)?;
    let tail = tail.trim();
    if !tail.is_empty() {
        return Err(TopicDdlError::Parse(format!(
            "unexpected trailing tokens: {tail}"
        )));
    }
    Ok(DropTopic { name, if_exists })
}

fn parse_column_list(text: &str) -> Result<Vec<Field>, TopicDdlError> {
    let mut fields: Vec<Field> = Vec::new();
    for raw in split_top_level_commas(text) {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        fields.push(parse_column_spec(raw)?);
    }
    if fields.is_empty() {
        return Err(TopicDdlError::Parse(
            "topic schema must declare at least one column".into(),
        ));
    }
    Ok(fields)
}

fn parse_column_spec(raw: &str) -> Result<Field, TopicDdlError> {
    let mut tokens = raw.split_whitespace();
    let name = tokens
        .next()
        .ok_or_else(|| TopicDdlError::Parse(format!("missing column name in `{raw}`")))?;
    let type_token = tokens
        .next()
        .ok_or_else(|| TopicDdlError::Parse(format!("missing column type in `{raw}`")))?;
    let mut not_null = false;
    while let Some(next) = tokens.next() {
        match next.to_uppercase().as_str() {
            "NOT" => {
                let next = tokens.next().ok_or_else(|| {
                    TopicDdlError::Parse(format!("`NOT` without `NULL` in `{raw}`"))
                })?;
                if !next.eq_ignore_ascii_case("NULL") {
                    return Err(TopicDdlError::Parse(format!(
                        "expected `NULL` after `NOT` in `{raw}`"
                    )));
                }
                not_null = true;
            }
            "NULL" => {
                // Explicit NULL is equivalent to the default (nullable).
            }
            other => {
                return Err(TopicDdlError::Parse(format!(
                    "unexpected token `{other}` in column spec `{raw}`"
                )));
            }
        }
    }
    let data_type = data_type_from_name(type_token).ok_or_else(|| {
        TopicDdlError::Parse(format!(
            "unsupported column type `{type_token}` in column spec `{raw}`"
        ))
    })?;
    Ok(Field::new(name, data_type, !not_null))
}

fn parse_with_options(text: &str) -> Result<BTreeMap<String, String>, TopicDdlError> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for raw in split_top_level_commas(text) {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let eq = raw
            .find('=')
            .ok_or_else(|| TopicDdlError::Parse(format!("WITH option missing `=` in `{raw}`")))?;
        let key = raw[..eq].trim().to_string();
        let value = unquote(raw[eq + 1..].trim());
        out.insert(key, value);
    }
    Ok(out)
}

fn unquote(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'') {
        value[1..value.len() - 1].replace("''", "'")
    } else {
        value.to_string()
    }
}

fn data_type_from_name(token: &str) -> Option<DataType> {
    match token.to_uppercase().as_str() {
        "INT" | "INTEGER" | "BIGINT" | "INT64" => Some(DataType::Int64),
        "FLOAT" | "DOUBLE" | "FLOAT64" => Some(DataType::Float64),
        "TEXT" | "UTF8" | "STRING" | "VARCHAR" => Some(DataType::Utf8),
        "BOOL" | "BOOLEAN" => Some(DataType::Boolean),
        _ => None,
    }
}

/// Strip an SQL keyword (case-insensitive) from the front of `s`, returning
/// the remainder. Returns `None` if the keyword is not present.
fn strip_keyword<'a>(s: &'a str, keyword: &str) -> Option<&'a str> {
    if s.len() >= keyword.len() && s[..keyword.len()].eq_ignore_ascii_case(keyword) {
        Some(&s[keyword.len()..])
    } else {
        None
    }
}

/// Consume the first SQL identifier from `s` — either a bare identifier
/// (letters / digits / underscores / dots) or a double-quoted identifier
/// — and return the unquoted name plus the remainder.
fn take_identifier(s: &str) -> Result<(String, &str), TopicDdlError> {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('"') {
        let end = rest
            .find('"')
            .ok_or_else(|| TopicDdlError::Parse("unterminated quoted identifier".into()))?;
        let name = rest[..end].to_string();
        Ok((name, &rest[end + 1..]))
    } else {
        let end = s
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
            .unwrap_or(s.len());
        if end == 0 {
            return Err(TopicDdlError::Parse("expected identifier".into()));
        }
        Ok((s[..end].to_string(), &s[end..]))
    }
}

/// Return the slice up to the matching closing paren and the remainder
/// after it. `s` is positioned just after the opening `(`. Handles nested
/// parens and single-quoted strings.
fn take_balanced_parens(s: &str) -> Result<(&str, &str), TopicDdlError> {
    let mut depth = 1;
    let mut in_string = false;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if c == b'\'' {
                // Handle the doubled-quote escape `''`.
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
        } else {
            match c {
                b'\'' => in_string = true,
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((&s[..i], &s[i + 1..]));
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    Err(TopicDdlError::Parse("unbalanced parens".into()))
}

/// Split `s` on top-level commas, ignoring commas inside parens or inside
/// single-quoted strings.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut depth = 0;
    let mut in_string = false;
    let mut start = 0;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if c == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
        } else {
            match c {
                b'\'' => in_string = true,
                b'(' => depth += 1,
                b')' if depth > 0 => depth -= 1,
                b',' if depth == 0 => {
                    out.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_for_non_topic_sql() {
        assert!(maybe_parse("SELECT 1").unwrap().is_none());
        assert!(maybe_parse("CREATE TABLE foo (id INT)").unwrap().is_none());
    }

    #[test]
    fn parses_create_topic_minimal() {
        let parsed = maybe_parse(
            "CREATE TOPIC cdc.orders (op TEXT NOT NULL, ts_ms BIGINT NOT NULL, key TEXT NOT NULL)",
        )
        .unwrap()
        .unwrap();
        let TopicDdl::Create(c) = parsed else {
            panic!("expected Create");
        };
        assert_eq!(c.name, "cdc.orders");
        assert_eq!(c.schema.fields().len(), 3);
        assert_eq!(c.schema.field(0).name(), "op");
        assert!(!c.schema.field(0).is_nullable());
        assert!(c.broker_metadata.is_empty());
    }

    #[test]
    fn parses_create_topic_with_options() {
        let parsed = maybe_parse(
            "CREATE TOPIC events (value FLOAT, payload TEXT) \
             WITH (retention_seconds = '86400', max_messages = '1000000')",
        )
        .unwrap()
        .unwrap();
        let TopicDdl::Create(c) = parsed else {
            panic!("expected Create");
        };
        assert_eq!(c.broker_metadata.get("retention_seconds").unwrap(), "86400");
        assert_eq!(c.broker_metadata.get("max_messages").unwrap(), "1000000");
    }

    #[test]
    fn parses_drop_topic() {
        let parsed = maybe_parse("DROP TOPIC cdc.orders").unwrap().unwrap();
        let TopicDdl::Drop(d) = parsed else {
            panic!("expected Drop");
        };
        assert_eq!(d.name, "cdc.orders");
        assert!(!d.if_exists);
    }

    #[test]
    fn parses_drop_topic_if_exists() {
        let parsed = maybe_parse("DROP TOPIC IF EXISTS cdc.orders")
            .unwrap()
            .unwrap();
        let TopicDdl::Drop(d) = parsed else {
            panic!("expected Drop");
        };
        assert!(d.if_exists);
    }

    #[test]
    fn rejects_create_topic_without_columns() {
        let err = maybe_parse("CREATE TOPIC empty ()").unwrap_err();
        assert!(err.to_string().contains("at least one column"));
    }

    #[test]
    fn rejects_create_topic_with_unknown_type() {
        let err = maybe_parse("CREATE TOPIC x (id WIDGET)").unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[test]
    fn handles_trailing_semicolon_and_whitespace() {
        let parsed = maybe_parse("  CREATE TOPIC foo (id INT) ;  ")
            .unwrap()
            .unwrap();
        assert!(matches!(parsed, TopicDdl::Create(_)));
    }
}
