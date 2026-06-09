//! Identifier quoting for generated SQL.
//!
//! A table, source, schema, or column name that the engine interpolates into a
//! generated SQL string is an *identifier*, not data: it must be ANSI
//! double-quoted (and any embedded `"` doubled) so a name carrying a hyphen,
//! mixed case, a space, or a reserved spelling resolves verbatim instead of
//! being re-parsed as an operator or keyword. Unquoted `patents-2024` parses as
//! `patents` minus `2024`; quoted `"patents-2024"` is the one identifier.
//!
//! Every site that builds SQL by string interpolation routes its identifiers
//! through here so the quoting rule lives in one place.

/// Quote a single SQL identifier with ANSI double-quotes, doubling any embedded
/// `"`. Accepted by DataFusion, SQLite, and Postgres alike.
///
/// ```
/// use jammi_db::sql::quote_ident;
/// assert_eq!(quote_ident("patents-2024"), r#""patents-2024""#);
/// assert_eq!(quote_ident(r#"a"b"#), r#""a""b""#);
/// ```
pub fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Quote a source-scoped table as the three-part relation a registered source's
/// data is addressed by: `"<source>".public."<table>"`. Both the source name
/// and the table name are quoted identifiers, so a hyphenated source (a user's
/// `patents-2024`) or table resolves verbatim.
pub fn source_relation(source_id: &str, table_name: &str) -> String {
    format!(
        "{}.public.{}",
        quote_ident(source_id),
        quote_ident(table_name)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_plain_identifier() {
        assert_eq!(quote_ident("papers"), r#""papers""#);
    }

    #[test]
    fn quotes_hyphenated_identifier() {
        // The #15 distributed-lane bug: an unquoted hyphen parses as a minus
        // operator. Quoting makes the hyphen part of the one identifier.
        assert_eq!(quote_ident("patents-2024"), r#""patents-2024""#);
    }

    #[test]
    fn doubles_embedded_quote() {
        assert_eq!(quote_ident(r#"a"b"#), r#""a""b""#);
    }

    #[test]
    fn source_relation_quotes_both_parts() {
        assert_eq!(
            source_relation("my-source-2024", "papers-v2"),
            r#""my-source-2024".public."papers-v2""#
        );
    }
}
